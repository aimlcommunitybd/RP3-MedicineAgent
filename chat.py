import sys
import argparse

from src.app.orchastrator import chat

parser = argparse.ArgumentParser(prog='cmd based chat for Medicine Agent')

from src.rag.ingest import ingest_data

def handle():
    # Ingest data
    ingest_data()

    parser.add_argument(
        'text', type=str, help='Handle for the Medicine Agent chat session'
    )
    args = parser.parse_args()
    response = chat(args.text)
    sys.stdout.write(response)
    return response


if __name__=="__main__":
    handle()