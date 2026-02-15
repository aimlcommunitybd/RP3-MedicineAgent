import json
from typing import Literal, Dict, Union, List
import structlog


from medicineagent.engine.slm_caller import generate_completion

logger = structlog.get_logger(__name__)


PURPOSE = "general"


def get_expert_advice(
    query: str,
    model: object,
    ground_knowledge: dict = None,
    context: str = None,
    previous_response: str = None,
    previous_response_evaluation: str = None,
    memory: List[dict] = None,
    max_tokens: int = 1024,
    temparature: float = 0.2,
    response_format: Literal["json", "text"] = "text",
    purpose: str = PURPOSE,
):
    logger.info(
        "Generating Expert Advice", query=query, ground_knowledge=ground_knowledge
    )

    knowledge_str = (
        str(ground_knowledge)
        if ground_knowledge
        else "No external knowledge available. Use your medical knowledge."
    )

    prompt = f"""You are an expert medical doctor. Answer the patient's query accurately and safely.

PATIENT QUERY: {query}

GROUND KNOWLEDGE (from web search):
{knowledge_str}

INSTRUCTIONS:
1. Only use the ground knowledge provided to answer medication-related questions
2. If ground knowledge is insufficient, state that clearly
3. Include warnings about drug interactions, side effects, and contraindications
4. Always recommend consulting a healthcare professional for medical advice
5. Be concise but complete
6. If you don't know something, say so honestly

Provide your expert response:"""

    if previous_response and previous_response_evaluation:
        prompt += f"""

PREVIOUS RESPONSE (which was rejected):
{previous_response}

EVALUATOR FEEDBACK:
{previous_response_evaluation}

Please address the evaluator's feedback and provide an improved response:"""

    completion, content = generate_completion(
        prompt=prompt,
        model=model,
        max_tokens=max_tokens,
        temparature=temparature,
        response_format=response_format,
        purpose=purpose,
    )
    logger.info("[Done] Expert Advice Generated", query=query, content=content)
    return content


def evaluate_expert_advice(
    query: str,
    junior_doctor_response: str,
    model: object,
    context: List[str] = None,
    ground_knowledge: dict = None,
    criteria: List[str] = None,
    max_tokens: int = 1024,
    temparature: float = 0.2,
    response_format: Literal["json", "text"] = "json",
    purpose: str = PURPOSE,
):
    logger.info(
        "Evaluating Expert Response",
        junior_doctor_response=junior_doctor_response,
        criteria=criteria,
    )
    criteria = criteria or [
        "Accuracy - Is the medical information correct?",
        "Relevance - Does it answer the patient's question?",
        "Clarity - Is it easy to understand?",
        "Completeness - Are all important aspects covered?",
        "Safety - Are there appropriate warnings?",
        "Use of Ground Knowledge - Does it cite the provided knowledge?",
    ]
    example_response = {
        "accuracy": 0.9,
        "relevance": 1.0,
        "clarity": 0.8,
        "completeness": 0.7,
        "safety": 0.9,
        "use_of_ground_knowledge": 0.8,
        "overall_score": 0.85,
        "key_issues": ["Issue 1", "Issue 2"],
        "strengths": ["Good point 1", "Good point 2"],
        "suggestions": ["Specific suggestion to improve"],
        "accept_response": False,
    }

    knowledge_str = (
        str(ground_knowledge) if ground_knowledge else "No ground knowledge provided."
    )

    prompt = f"""You are a Senior Medical Expert reviewing a junior doctor's response. Be STRICT - medical accuracy is critical for patient safety.

PATIENT QUERY: {query}

JUNIOR DOCTOR'S RESPONSE:
{junior_doctor_response}

GROUND KNOWLEDGE PROVIDED:
{knowledge_str}

EVALUATION CRITERIA:
{chr(10).join(f"- {c}" for c in criteria)}

IMPORTANT:
1. If there's no ground knowledge, check if the response appropriately mentions this limitation
2. For medication questions, verify drug names, dosages, and interactions are correct
3. Flag any potential safety concerns or missing warnings
4. A response should ONLY be accepted if it's accurate, safe, and answers the question
5. Be more likely to reject than accept - patient safety comes first

Provide your evaluation in JSON format:
{json.dumps(example_response)}

Remember: Provide your response in JSON format only. Open and Close with curly braces."""

    completion, content = generate_completion(
        prompt=prompt,
        model=model,
        max_tokens=max_tokens,
        temparature=temparature,
        response_format=response_format,
        purpose=purpose,
        example_response=example_response,
    )

    logger.info(
        "[Done] Expert Response Evaluated",
        junior_doctor_response=junior_doctor_response,
        evaluator_doctor_response=content,
    )
    return content
