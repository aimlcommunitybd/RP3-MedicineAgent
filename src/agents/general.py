import time
from typing import Literal, Dict, Union, List

import structlog

from src.prompts import general as template
from src.engine.slm_caller import generate_completion

PURPOSE = "general"
logger = structlog.get_logger(__name__)


def classify_text(
    text: str,
    model: Union[object, str],
    context: str = None,
    max_tokens: int = 100,
    temparature: float = 0.8,
    response_format: Literal["json", "text"] = "json",
    purpose: str = PURPOSE,
):
    prompt = template.classification.format(
        query=text,
        context=context,
        team_expertise=template.team_expertise,
        identity=template.admin_identity,
        response_format=response_format,
        # classes=template.classification_classes,
        example_response=template.classification_example_response,
    )
    logger.info(
        "Classifying Givent Query",
        prompt=prompt,
        text=text,
        model=model,
        response_format=response_format,
    )
    completion, content = generate_completion(
        prompt=prompt,
        model=model,
        max_tokens=max_tokens,
        temparature=temparature,
        response_format=response_format,
        purpose=purpose,
        # example_reponse=template.classification_example_response,
    )
    logger.info("[Done] Classification", query=text, content=content)
    return content


def response_irrelevent_query(
    text: str,
    model: Union[object, str],
    context: str = None,
    query_class: str = "General Query",
    max_tokens: int = 100,
    temparature: float = 0.8,
    response_format: Literal["json", "text"] = "text",
    purpose: str = PURPOSE,
):
    prompt = template.irrelevent_query_response.format(
        query=text,
        query_class=query_class,
        identity=template.assistant_identity,
    )
    logger.info(
        "Generating Answer for Irrelevent Query",
        query=text,
        prompt=prompt,
        context=context,
        purpose=purpose,
        model=model,
    )
    completion, content = generate_completion(
        prompt=prompt,
        model=model,
        max_tokens=max_tokens,
        temparature=temparature,
        response_format=response_format,
        purpose=purpose,
        # example_reponse=template.classification_example_response,
    )
    logger.info(
        "[Done] Generated answer for irrelevent query", query=text, content=content
    )
    return content


def generate_relevent_response(
    text: str,
    model: Union[object, str],
    context: str = None,
    max_tokens: int = 100,
    temparature: float = 0.8,
    response_format: Literal["json", "text"] = "text",
    purpose: str = PURPOSE,
):
    prompt = template.general_relevent_response.format(
        query=text,
        context=context,
        identity=template.doctor_identity,
    )
    logger.info(
        "Generating relevent query response",
        query=text,
        context=context,
        model=model,
        prompt=prompt,
    )
    completion, content = generate_completion(
        prompt=prompt,
        model=model,
        max_tokens=max_tokens,
        temparature=temparature,
        response_format=response_format,
        purpose=purpose,
        # example_reponse=template.classification_example_response,
    )
    logger.info("[Done] Generated relevent query answer", query=text, content=content)
    return content

def extract_named_entity(
    text: str,
    model: Union[object, str],
    context: str = None,
    query_class: str = "General Query",
    max_tokens: int = 100,
    temparature: float = 0.8,
    response_format: Literal["json", "text"] = "json",
    purpose: str = PURPOSE,
):
    prompt = "given a user query, check if there is any medicine name. if yes, extract the medicine name and return in json format as {'medicine_names': [list of names]}. if no medicine name is present, return {'medicine_names': []}\n\nUser Query: {text}\n\nResponse:"
    logger.info(
        "Extracting Named Entities from Query",
        query=text,
        prompt=prompt,
    )
    completion, content = generate_completion(
        prompt=prompt,
        model=model,
        max_tokens=max_tokens,
        temparature=temparature,
        response_format=response_format,
        purpose=purpose,
        # example_reponse=template.classification_example_response,
    )
    logger.info("[Done] Named Entity Extraction", query=text, content=content)
    return content

def find_generic_name(
    medicine_names: List[str],
    model: Union[object, str],
    max_tokens: int = 200,
    temparature: float = 0.8,
    response_format: Literal["json", "text"] = "json",
    purpose: str = PURPOSE,
):
    logger.info("Finding Generic Names for Medicines", medicine_names=medicine_names)
    prompt = "Given a list of medicine names, return their generic names in a JSON format as {'medicines': {'name': 'generic_name', ...}}. If a medicine name does not have a known generic name, return 'empty str' for that entry.\n\nMedicine Names: {medicine_names}\n\nResponse:"
    completion, content = generate_completion(
        prompt=prompt,
        search_prompt=f"What are the generic names for the following medicines: {medicine_names}?",
        model=model,
        max_tokens=max_tokens,
        temparature=temparature,
        response_format=response_format,
        purpose=purpose,
        # example_reponse=template.classification_example_response,
    )
    logger.info("[Done] Finding Generic Names for Medicines", medicine_names=medicine_names, content=content)
    knowledge = completion.choices[0].message.annotations
    return content, knowledge  

def find_drug_to_drug_interaction(
    generic_names:List[str],
    knowledge: Dict,
    model: Union[object, str],
    max_tokens: int = 300,
    temparature: float = 0.8,
    response_format: Literal["json", "text"] = "text",
    purpose: str = PURPOSE,
):
    prompt = "Given a list of generic medicine names, check if there are any known drug-to-drug interactions among them. Return the interactions summary.\n\nGeneric Medicine Names: {generic_names}\n\nWeb Knowledge: {knowledge}\n\nResponse:"
    logger.info("Finding Drug-to-Drug Interactions", generic_names=generic_names)
    completion, content = generate_completion(
        prompt=prompt,
        search_prompt=f"Are there any drug-to-drug interactions among the following medicines: {generic_names}?",
        model=model,
        max_tokens=max_tokens,
        temparature=temparature,
        response_format=response_format,
        purpose=purpose,
        # example_reponse=template.classification_example_response,
    )
    logger.info("[Done] Finding Drug-to-Drug Interactions", generic_names=generic_names, content=content)
    knowledge = completion.choices[0].message.annotations
    return content, knowledge
    

def rewrite_empathic_response(
    text: str,
    model: Union[object, str],
    context: str = None,
    max_tokens: int = 150,
    temparature: float = 0.8,
    response_format: Literal["json", "text"] = "text",
    purpose: str = PURPOSE,
):
    prompt = f"""You are a compassionate and empathetic medical assistant. Rewrite the following response to ensure it is delivered with empathy and understanding, while maintaining accuracy and clarity.\n\nOriginal Response: '{text}'\n\nContext: {context}\n\nProvide the rewritten response.
    """
    logger.info(
        "Rewriting Response with Empathy",
        query=text,
        context=context,
        model=model,
        prompt=prompt,
    )
    completion, content = generate_completion(
        prompt=prompt,
        model=model,
        max_tokens=max_tokens,
        temparature=temparature,
        response_format=response_format,
        purpose=purpose,
        # example_reponse=template.classification_example_response,
    )
    logger.info("[Done] Empathic Response Rewritten", query=text, content=content)
    return content

def reply_general_query(
    text: str,
    model: Union[object, str],
    chat_history: Union[List[dict], None] = None,
    max_tokens: int = 300,
    temparature: float = 0.8,
    response_format: Literal["json", "text"] = "text",
    purpose: str = PURPOSE,
):
    prompt = """Given a query from a user, provide a detailed and accurate response based on your medical knowledge and expertise. Ensure that the response is clear, concise, and easy to understand for the user. \n\nUser Query: '{text}'\n\nContext: {chat_history}\n\nResponse:"""
    
    logger.info(
        "Generating General Query Response",
        query=text,
        context=chat_history,
        model=model,
        prompt=prompt,
    )
    completion, content = generate_completion(
        prompt=prompt,
        model=model,
        max_tokens=max_tokens,
        temparature=temparature,
        response_format=response_format,
        purpose=purpose,
        # example_reponse=template.classification_example_response,
    )
    logger.info("[Done] General Query Response Generated", query=text, content=content)
    return content