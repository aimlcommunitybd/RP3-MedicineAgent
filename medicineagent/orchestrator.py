from typing import Union, List, Dict
import time
from datetime import datetime

import structlog
from llama_cpp import Llama

from medicineagent.agents.general import *
from medicineagent.agents.expert import *
from medicineagent.engine.slm_caller import generate_completion
from medicineagent.memory import PersistentChatHistory

logger = structlog.get_logger(__name__)


def chat(
    text: str,
    general_model: Union[object, str] = None,
    expert_model: Union[object, str] = None,
    chat_history: Union[List[dict], None] = None,
) -> str:
    # Classification
    strtime = time.perf_counter()
    datetime_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    global_chat_history = PersistentChatHistory(
        f"medicineagent/memory/chat_history_{datetime_now}.json"
    )
    global_chat_history.append(chat_history[-1] if chat_history else {})

    result = classify_text(text=text, model=general_model)
    is_relevant = result["result"]
    if not is_relevant:
        reply = response_irrelevent_query(
            text,
            model=general_model,
        )
        global_chat_history.append({"role": "assistant", "content": reply})
        return format_reply(query=text, reply=reply, strtime=strtime)

    # NER
    named_entity = extract_named_entity(text=text, model=general_model)
    if named_entity and named_entity.get("medicine_names"):
        medicine_list = named_entity["medicine_names"]
        generic_names_result, search_result = find_generic_name(
            medicine_names=medicine_list, model=general_model
        )

        # Handle case where JSON parsing failed
        if isinstance(generic_names_result, dict) and "error" in generic_names_result:
            logger.warning(
                "Generic name lookup failed, using medicine names directly",
                error=generic_names_result,
            )
            # Use the original medicine names as fallback
            generic_names = medicine_list
        elif (
            isinstance(generic_names_result, dict)
            and "medicines" in generic_names_result
        ):
            generic_names = list(generic_names_result.get("medicines", {}).values())
        else:
            generic_names = medicine_list  # Fallback

        # Find D2C Interactions
        # Websearch
        interaction_summary, search_result = find_drug_to_drug_interaction(
            generic_names=generic_names, knowledge=search_result, model=general_model
        )
        # RAG
        # TBA
    else:
        reply = reply_general_query(
            text=text, model=general_model, chat_history=chat_history
        )
        global_chat_history.append({"role": "assistant", "content": reply})
        return format_reply(query=text, reply=reply, strtime=strtime)

    # Expert Response Generation
    # Supervisor based Response Validation
    # Empathy rewrite
    is_deliverable_response = False
    retry_count = 0
    previous_response = None
    previous_response_evaluation = None
    expert_response = None

    while not is_deliverable_response and retry_count < 3:
        logger.info(
            f"Generating Expert Response [retry {retry_count}]", retry_count=retry_count
        )
        expert_response = get_expert_advice(
            query=text,
            model=expert_model,
            ground_knowledge=search_result if search_result else {},
            previous_response=previous_response,
            previous_response_evaluation=previous_response_evaluation,
        )
        if not expert_response or len(expert_response.strip()) < 10:
            logger.warning(
                f"Expert response too short or empty, retrying...",
                expert_response=expert_response,
            )
            retry_count += 1
            continue
        global_chat_history.append({"role": "expert", "content": expert_response})
        try:
            evaluation = evaluate_expert_advice(
                query=text,
                junior_doctor_response=expert_response,
                model=expert_model,
                ground_knowledge=search_result if search_result else {},
            )
            if isinstance(evaluation, dict) and "accept_response" in evaluation:
                global_chat_history.append(
                    {"role": "evaluator", "content": str(evaluation)}
                )
                # Store for next iteration feedback
                previous_response = expert_response
                previous_response_evaluation = evaluation.get(
                    "suggestions", []
                ) + evaluation.get("key_issues", [])
                previous_response_evaluation = (
                    str(previous_response_evaluation)
                    if previous_response_evaluation
                    else str(evaluation)
                )

                if evaluation["accept_response"]:
                    is_deliverable_response = True
                else:
                    logger.info("Response rejected, will retry", evaluation=evaluation)
                retry_count += 1
            else:
                logger.warning(
                    "Evaluation response missing accept_response, treating as not accepted",
                    evaluation=evaluation,
                )
                retry_count += 1
        except Exception as e:
            logger.error(f"Error in evaluation: {e}")
            retry_count += 1

    # Fallback if all retries failed
    if not expert_response:
        expert_response = "I apologize, but I couldn't generate a reliable response. Please consult a healthcare professional for accurate medical advice."

    # return generate_relevent_response(text, general_model)
    reply = rewrite_empathic_response(
        text=expert_response,
        model=general_model,
        context=chat_history,
    )
    global_chat_history.append({"role": "assistant", "content": reply})
    return format_reply(query=text, reply=reply, strtime=strtime)
    # Return

    # Time and Resource Tracker
    # Shared Context Memory
    # ChatID based Context tracker

    # Chat UI (future)
    return print(f"Hard Coded Response: {text}")


def format_reply(query: str, reply: str, strtime: float = None) -> Dict[str, str]:
    result = {"user": query, "chatbot": reply}
    if strtime is not None:
        result["time_taken"] = time.perf_counter() - strtime
    return result
