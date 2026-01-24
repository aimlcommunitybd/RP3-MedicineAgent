import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.engine import openrouter


def handle():
    parser = argparse.ArgumentParser(
        prog="Test OpenRouter Generation",
        description="Test text generation using OpenRouter API",
    )
    parser.add_argument("prompt", type=str, help="prompt for model")
    parser.add_argument(
        "--model",
        default=openrouter.MODEL,
        help="model name. You can use other models by writing other models name from openrouter.RouterModel. eg. --model 'meta-llama/llama-3.2-3b-instruct:free'",
    )
    args = parser.parse_args()
    completion, content = openrouter.api_complete(prompt=args.prompt, model=args.model)
    sys.stdout.write(content)
    sys.stdout.write("\n")
    return completion, content


if __name__ == "__main__":
    # usage: uv run scripts/test_openrouter.py "what is todays date"
    # usage with custom model: uv run scripts/test_openrouter.py "what is todays date" --model "meta-llama/llama-3.2-3b-instruct:free"
    handle()
