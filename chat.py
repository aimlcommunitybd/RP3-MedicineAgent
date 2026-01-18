import sys
import argparse

from src.app.orchastrator import chat

parser = argparse.ArgumentParser(prog='cmd based chat for Medicine Agent')

def handle():
    parser.add_argument(
        'text', type=str, help='Handle for the Medicine Agent chat session'
    )
    args = parser.parse_args()
    response = f"\n\nHardcoded Dummy Response: {args.text}\n\n"
    sys.stdout.write(response)
    return response


if __name__=="__main__":
    handle()