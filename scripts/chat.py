import sys
import argparse
from typing import Union
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from medicineagent import settings
from medicineagent.engine.llamacpp import load_gguf_model
from medicineagent.engine.openrouter import MODEL as OpenRouterModel
from medicineagent.orchestrator import chat

GENERAL_MODEL = OpenRouterModel
EXPERT_MODEL = OpenRouterModel  # FIXME: load_gguf_model(model_path=settings.EXPERT_MODEL_PATH)


def handle():
    parser = argparse.ArgumentParser(prog="cmd based chat for Medicine Agent")
    parser.add_argument("--query", type=str, help="Pass for single query and answer")
    parser.add_argument(
        "--live_chat",
        action="store_true",
        help="Pass this flag for interactive chat session",
    )
    args = parser.parse_args()

    # General Single Query
    if not args.live_chat:
        response = chat(
            text=args.query,
            chat_history=None,
            expert_model=EXPERT_MODEL,
            general_model=OpenRouterModel,
        )
        sys.stdout.write(f"ChatBot: {response}\n")

    # Live Chat Session
    else:
        run_live_chat(
            general_model=GENERAL_MODEL,
        )


def run_live_chat(
    expert_model: Union[object, str] = None,
    general_model: Union[object, str] = GENERAL_MODEL,
):
    chat_history = []
    sys.stdout.write(
        f"\n{'==='*80}\nTurning on Interactive Chat Session. Press 'ctrl+c' to stop.\n{'==='*80}\n\n"
    )
    try:
        while True:
            query = input("User: ")
            response = chat(
                text=query,
                chat_history=chat_history,
                expert_model=expert_model,
                general_model=general_model,
            )
            message = [
                {"role": "user", "content": query},
                {"role": "assistant", "content": response},
            ]
            chat_history.extend(message)
            sys.stdout.write(f"ChatBot: {response}\n")
    except KeyboardInterrupt:
        sys.stdout.write(
            f"\n{'==='*80}\n\nInteractive Chat Session Closed\n\n{'==='*80}\n"
        )  # Fixed
        sys.exit()
    except Exception as exc:
        raise exc


if __name__ == "__main__":
    # Usage for cotinuous chat: uv run chat.py --live_chat
    # Usage for single query: uv run chat.py --query "What is Type-2 Diabetes?"
    handle()
