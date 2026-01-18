import os
import argparse
from pathlib import Path

import structlog
from huggingface_hub import hf_hub_download

logger = structlog.get_logger(__name__)


def download_model(
    repo_id: str,
    filename: str,
    save_dir: Path = Path("./src/models/downloads/"),
    repo_type: str = "model",
) -> None:
    """
    Download a model file from Hugging Face Hub.
    Args:
        repo_id (str): huggingface repo id
        filename (str): model file name to download
        save_dir (Path): directory to save the downloaded model. Default to "./src/models/downloads/"
        repo_type (str, optional): type of repository. Defaults to "model".
    """
    os.makedirs(save_dir, exist_ok=True)
    logger.info(
        "Downloading model from Hugging Face",
        repo_id=repo_id,
        model_file=filename,
        save_dir=save_dir,
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=save_dir,
        repo_type=repo_type,
    )
    logger.info(
        "Model downloaded successfully", model_path=model_path, save_dir=save_dir
    )


def main():
    """
    usage: 
        python download_model.py taufiq-ai/qwen2.5-coder-1.5-instruct-ft qwen2.5-coder-1.5b-instruct-mt-04092025-v2.gguf --save_dir src/models/
    """
    parser = argparse.ArgumentParser(description="Download model from Hugging Face")
    parser.add_argument(
        "repo_id", type=str, help="Hugging Face repository ID"
    )
    parser.add_argument(
        "filename", type=str, help="Model file name to download",
    )
    parser.add_argument(
        "--save_dir", type=str, default="./src/models/downloads/", help="Directory to save the downloaded model",
    )
    parser.add_argument(
        "--repo_type", type=str, default="model", help="Type of HuggingFace repositoy"
    )
    args = parser.parse_args()
    download_model(
        repo_id=args.repo_id, filename=args.filename, save_dir=Path(args.save_dir), repo_type=args.repo_type,
    )


if __name__ == "__main__":
    main()
