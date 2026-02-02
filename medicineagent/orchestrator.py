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
    global_chat_history = PersistentChatHistory(f"medicineagent/memory/chat_history_{datetime_now}.json")
    global_chat_history.append(chat_history[-1] if chat_history else {})
    
    result = classify_text(text=text, model=general_model)
    is_relevant = result["result"]
    if not is_relevant:
        reply = response_irrelevent_query(
            text, model=general_model,
        )
        global_chat_history.append({"role": "assistant", "content": reply})
        return format_reply(query=text, reply=reply, strtime=strtime)
    
    # NER
    named_entity = extract_named_entity(text=text, model=general_model)
    if named_entity:
        generic_names, search_result = find_generic_name(medicine_names=named_entity["medicine_names"], model=general_model)
        # Find D2C Interactions
        # Websearch
        interaction_summary, search_result = find_drug_to_drug_interaction(
            generic_names=generic_names, knowledge=search_result, model=general_model
        )
        # RAG
        # TBA
    else:
        reply = reply_general_query(text=text, model=general_model, chat_history=chat_history)
        global_chat_history.append({"role": "assistant", "content": reply})
        return format_reply(query=text, reply=reply, strtime=strtime)
    
    # Expert Response Generation
    # Supervisor based Response Validation
    # Empathy rewrite
    # FIXME: For now use general model
    is_deliverable_response = False
    retry_count = 0
    while not is_deliverable_response and retry_count < 3:
        logger.info(f"Generating Expert Response [{retry_count}]", retry_count=retry_count)
        previous_response = None
        previous_response_evaluation = None
        expert_response = get_expert_advice(
            query=text, 
            model=expert_model, ground_knowledge=search_result, previous_response=previous_response, previous_response_evaluation=previous_response_evaluation
        )
        global_chat_history.append({"role": "expert", "content": expert_response})
        evaluation = evaluate_expert_advice(
            query=text,
            junior_doctor_response=expert_response,
            model=expert_model,
            ground_knowledge=search_result,
        ) 
        global_chat_history.append({"role": "evaluator", "content": evaluation})
        previous_response = expert_response
        previous_response_evaluation = evaluation
        retry_count += 1
        if evaluation["accept_response"]:
            is_deliverable_response = True

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


def format_reply(query: str, reply: str, strtime:float=None) -> Dict[str, str]:
    result = {
        "user": query,
        "chatbot": reply
    }
    if strtime is not None:
        result["time_taken"] = time.perf_counter() - strtime
    return result