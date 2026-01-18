import sys
import argparse

from app.orchastrator import chat

parser = argparse.ArgumentParser(prog='cmd based chat for Medicine Agent')

def handle():
    parser.add_argument(
        'text', type=str, help='Handle for the Medicine Agent chat session'
    )
    args = parser.parse_args()
    response = chat(args.text)
    sys.stdout.write(f"Response: {response}\n\n")
    return args.text

if __name__=="__main__":
    handle()