import time
from typing import Literal, Dict, Union, List

import structlog

from medicineagent.prompts import general as template
from medicineagent.engine.slm_caller import generate_completion

PURPOSE = "general"
logger = structlog.get_logger(__name__)


def classify_text(
    text: str,
    model: Union[object, str],
    context: str = None,
    max_tokens: int = 256,
    temparature: float = 0.3,
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
    max_tokens: int = 256,
    temparature: float = 0.3,
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
    max_tokens: int = 512,
    temparature: float = 0.3,
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
    max_tokens: int = 256,
    temparature: float = 0.3,
    response_format: Literal["json", "text"] = "json",
    purpose: str = PURPOSE,
):
    prompt = f"""Extract ALL medicine/drug names from the user query, including brand names, generic names, and combination drugs. Look for patterns like:
- Numbers in medicine names (e.g., Napa500, Fymoxil500, Napa, Fymoxil, Napa+)
- Common drug suffixes (-cilin, -ox, -ine, -ol, -in, -ate)
- Any words that could be medication names

User Query: {text}

Return in JSON format:
{{"medicine_names": ["list of all medicine names found"]}}

If no medicines found, return {{"medicine_names": []}}"""

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
    max_tokens: int = 512,
    temparature: float = 0.3,
    response_format: Literal["json", "text"] = "json",
    purpose: str = PURPOSE,
):
    logger.info("Finding Generic Names for Medicines", medicine_names=medicine_names)

    example_response = {
        "medicines": {"Aspirin": "Acetylsalicylic acid", "Panadol": "Paracetamol"}
    }

    prompt = f"""Given a list of medicine names, return their generic names in a JSON format as {{"medicines": {{"brand_name": "generic_name"}}}}.

Common medicine references:
- Napa500, Napa, Paracetamol = Paracetamol (Acetaminophen)
- Fymoxil500, Fymoxil = Amoxicillin with Clavulanic acid
- Napa Plus = Paracetamol + Tramadol
- Monocef = Ceftriaxone
- Napa Extend = Paracetamol extended release

Medicine Names: {medicine_names}

Response:"""

    completion, content = generate_completion(
        prompt=prompt,
        search_prompt=f"What are the generic names for: {', '.join(medicine_names)}? Include brand name and generic name.",
        model=model,
        max_tokens=max_tokens,
        temparature=temparature,
        response_format=response_format,
        purpose=purpose,
        example_response=example_response,
    )
    logger.info(
        "[Done] Finding Generic Names for Medicines",
        medicine_names=medicine_names,
        content=content,
    )
    knowledge = (
        completion.choices[0].message.annotations
        if hasattr(completion, "choices")
        else None
    )
    return content, knowledge


def find_drug_to_drug_interaction(
    generic_names: List[str],
    knowledge: Dict,
    model: Union[object, str],
    max_tokens: int = 512,
    temparature: float = 0.3,
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
    logger.info(
        "[Done] Finding Drug-to-Drug Interactions",
        generic_names=generic_names,
        content=content,
    )
    knowledge = completion.choices[0].message.annotations
    return content, knowledge


def rewrite_empathic_response(
    text: str,
    model: Union[object, str],
    context: str = None,
    max_tokens: int = 512,
    temparature: float = 0.3,
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
    max_tokens: int = 512,
    temparature: float = 0.3,
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
