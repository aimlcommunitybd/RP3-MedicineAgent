import os 

import structlog
from llama_cpp import Llama

from src import settings

logger = structlog.get_logger(__name__)



def load_model(
    model_path: str = settings.GENERAL_MODEL_PATH,
    context_window: int = settings.CONTEXT_WINDOW,
) -> Llama:
    """
    load gguf formatted model from local
    """
    if not os.path.exists(model_path):
        logger.error(
            f"Recheck  MODEL_PATH': '{model_path}' is correct path",
            model_path=model_path,
        )
        raise FileNotFoundError(f"Model file not found at {model_path}")

    logger.info(f"Loading model", context_window=context_window, model_path=model_path)
    model = Llama(
        model_path=model_path,
        n_ctx=context_window,
    )
    return model

# def generate_response(prompt:Union[str, dict],modelpath:str="models/medgemma-4b-it"):
#     processor = AutoProcessor.from_pretrained(modelpath)
#     model = AutoModelForImageTextToText.from_pretrained(modelpath)
#     if isinstance(prompt, str):
#         messages = [
#             {
#                 "role": "user",
#                 "content": [
#                     # {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
#                     {"type": "text", "text": prompt}
#                 ]
#             },
#         ]
#     else: messages = prompt
#     inputs = processor.apply_chat_template(
#         messages,
#         add_generation_prompt=True,
#         tokenize=True,
#         return_dict=True,
#         return_tensors="pt",
#     ).to(model.device)
#     outputs = model.generate(**inputs, max_new_tokens=40)
#     content = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:])
#     return content