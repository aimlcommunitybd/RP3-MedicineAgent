from typing import Literal, Dict, Union, List
import structlog


from medicineagent.engine.slm_caller import generate_completion

logger = structlog.get_logger(__name__)


# PURPOSE = "expert"
PURPOSE = "general"

def get_expert_advice(
    query: str,
    model: object,
    ground_knowledge: dict = None,
    context: str = None,
    previous_response: str = None,
    previous_response_evaluation: str = None,
    memory: List[dict] = None,
    max_tokens: int = 500,
    temparature: float = 0.7,
    response_format: Literal["json", "text"] = "text",
    purpose: str = PURPOSE,
):
    logger.info("Generating Expert Advice", query=query, ground_knowledge=ground_knowledge)
    
    prompt = f"""You are an expert in medicine and healthcare. Based on the query '{query}', provide expert advice using the following ground knowledge: {ground_knowledge}. \n\nContext: {context}\n\nProvide a detailed and accurate response."""
    if previous_response_evaluation:
        prompt += f"\n\nNote: Givent evaluation of your previous responses:. Please improve based on this feedback. \n\nYour Previous Response: {previous_response}\n\nEvaluation: {previous_response_evaluation}"
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
    max_tokens: int = 1000,
    temparature: float = 0.5,
    response_format: Literal["json", "text"] = "json",
    purpose: str = PURPOSE,
):
    logger.info("Evaluating Expert Response", junior_doctor_response=junior_doctor_response, criteria=criteria)
    criteria = criteria or [
        "Accuracy",
        "Relevance",
        "Clarity",
        "Completeness",
        "Use of Ground Knowledge",
    ]
    example_response = {
        "accuracy": 0.8,
        "relevance": 0.9,
        "clarity": 0.85,
        "completeness": 0.75,
        "use_of_ground_knowledge": 0.9,
        "overall_score": 0.84,
        "comments": "Brief summary of key strengths and weaknesses",
        "suggestions": ["Suggestion 1", "Suggestion 2"],
        "accept_response": True
    }
    
    prompt = f"""You are an Senior Expert in Medicine. Evaluate the following response is correct based on patients query and medical knowledge. \n\nInitial junior_doctor_response: '{junior_doctor_response}'. \n\nEvaluation Criteria: {criteria}\n\nProvide a wise evaluation. grounded knowledge: {ground_knowledge}.\n\nContext: {context}.\n\nProvide your evaluation in Json format. \n\nExample response: {example_response}\n\nRemember: Provide your response in Json format only. Open and Close with curly braces."""
    
    completion, content = generate_completion(
        prompt=prompt,
        model=model,
        max_tokens=max_tokens,
        temparature=temparature,
        response_format=response_format,
        purpose=purpose,
        example_response=example_response
    )
    
    logger.info("[Done] Expert Response Evaluated", junior_doctor_response=junior_doctor_response, evaluator_doctor_response=content)
    return content
    
    
