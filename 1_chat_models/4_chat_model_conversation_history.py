from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables from a .env file (API keys, etc.).
load_dotenv()

# Conversation history stored as a list of Message objects
chat_history: list = []

# System prompt that defines assistant behavior. Important: do NOT leave a
# trailing comma here — that would create a tuple instead of a SystemMessage.
system_message = SystemMessage(content="You are a helpful assistant.")
chat_history.append(system_message)

# Initialize the LLM client. Adjust model_name and temperature as needed.
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)


def print_history(history: list) -> None:
    """Pretty-print the conversation history.

    Prints each message with an index and role label so it's easy to read.
    """
    print("\n-------- Message History --------")
    for i, msg in enumerate(history, start=1):
        if isinstance(msg, SystemMessage):
            role = "System"
        elif isinstance(msg, HumanMessage):
            role = "Human"
        elif isinstance(msg, AIMessage):
            role = "AI"
        else:
            role = type(msg).__name__
        # Some Message objects expose their text as `.content` — print safely.
        content = getattr(msg, "content", repr(msg))
        print(f"{i}. {role}: {content}")
    print("---------------------------------\n")


def main_loop() -> None:
    """Main interactive loop: read user input, send history to LLM, print reply."""
    while True:
        query = input("Human/You: ")
        if query.strip().lower() in {"quit", "exit"}:
            print("Exiting conversation.")
            break

        # Append the user's message to the history
        chat_history.append(HumanMessage(content=query))

        # Invoke the model with the full conversation history so it has context
        result = llm.invoke(chat_history)

        # Extract response content and append to history
        response = getattr(result, "content", str(result))
        chat_history.append(AIMessage(content=response))

        # Print the assistant's reply and the updated history
        print(f"AI: {response}\n")
        print_history(chat_history)


if __name__ == "__main__":
    main_loop()