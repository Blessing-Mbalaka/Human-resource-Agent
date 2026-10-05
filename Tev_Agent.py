import argparse

import httpx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from ollama import ResponseError

from workflow import run_workflow


MODEL_NAME = "llama3.2:1b"
SCREENING_COMMANDS = {"screen", "screen applicants", "start screening"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Applicant screening console prototype.")
    parser.add_argument(
        "--chat",
        action="store_true",
        help="start the conversational console instead of screening the CV folder",
    )
    args = parser.parse_args()
    if not args.chat:
        run_workflow()
        return

    _run_chat()


def _run_chat() -> None:
    chat = ChatOllama(model=MODEL_NAME, temperature=0.3)
    history: list[HumanMessage | AIMessage] = []

    print("Hello! We can chat or run an applicant screening batch.")
    print("Type 'screen applicants' to process the CV folder, 'help' for commands, or 'quit'.")

    while True:
        try:
            user_text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return

        command = user_text.casefold()
        if command in {"quit", "exit"}:
            print("Goodbye.")
            return
        if command == "help":
            print("Commands: screen, help, quit. Otherwise, chat normally.")
            continue
        if command in SCREENING_COMMANDS:
            try:
                run_workflow()
            except (OSError, ValueError, RuntimeError) as error:
                print(f"Workflow error: {error}")
            continue
        if not user_text:
            continue

        messages = [
            SystemMessage(
                content=(
                    "Be friendly and concise in small talk. Do not evaluate "
                    "applicants or make hiring decisions. Direct screening to "
                    "the separate screen command."
                )
            ),
            *history,
            HumanMessage(content=user_text),
        ]
        try:
            answer = chat.invoke(messages)
        except (ConnectionError, TimeoutError, httpx.HTTPError, ResponseError) as error:
            print(f"Could not get a response from Ollama: {error}")
            print("Check that Ollama is running and the model is available.")
            continue

        answer_text = str(answer.content)
        print(f"Assistant: {answer_text}")
        history.extend(
            [HumanMessage(content=user_text), AIMessage(content=answer_text)]
        )


if __name__ == "__main__":
    main()
