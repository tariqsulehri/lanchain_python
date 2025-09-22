from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory
import os
from dotenv import load_dotenv

# Load environment variables from a .env file (API keys, etc.).
load_dotenv()
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_TRACE"] = "" # Comma-separated list of components to log

PROJECT_ID = "gpt5py"  # Replace with your Firebase project ID
SESSION_ID = "user_session_new"  # Unique ID for this chat session
COLLECTION_NAME = "chat_history"  # Firestore collection to store chat histories


print("Initializing Firestore client...")
client = firestore.Client(project=PROJECT_ID)

print("Initlizing Firestore Chat Message History...")
chat_history = FirestoreChatMessageHistory(
    session_id=SESSION_ID,
    collection=COLLECTION_NAME,
    client=client,
)

print("Firestore Chat Message History initialized.")
print("Current chat history loaded from Firestore:", chat_history.messages)

# Firebase/Firestore setup notes (high-level, no code):
# 1) Create a Firebase project
#    - Go to the Firebase Console and create a new project.
#    - Choose a project name and (optionally) link a Google Analytics account.
#
# 2) Enable Firestore in the project
#    - In the Firebase Console navigate to "Firestore Database".
#    - Create a Firestore database and choose a location/region.
#    - Pick the desired security rules mode (Start in test mode for development,
#      then switch to locked mode and write rules before production).
#
# 3) Create a service account (for server-side access)
#    - In the Firebase project settings, open "Service accounts".
#    - Generate a new private key for a service account and download the JSON key
#      file. Keep this file secure; it provides admin access to your Firestore.
#
# 4) Store credentials securely for your app
#    - Never check the service account JSON into source control.
#    - For local development, point to the JSON via an environment variable or
#      load it securely from a secrets manager. In production, use environment
#      variables or the hosting platform's secret manager (e.g., Cloud Run,
#      App Engine, or GitHub Actions secrets).
#
# 5) Design a Firestore data model for chat history
#    - Typical pattern: a `conversations` collection with documents for each
#      conversation (document id can be a UUID). Under each conversation
#      document, store a `messages` subcollection where each message is a
#      document containing: role (system/human/ai), content, timestamp, and any
#      metadata (e.g., model used, message id).
#    - Alternatively, store messages as an array field on the conversation
#      document for small histories, but prefer a subcollection for growth.
#
# 6) Set Firestore security rules
#    - Write rules that limit who can read/write conversations. For example,
#      allow authenticated users to create their own conversations and read only
#      their own documents. If using server-side writes (service account), keep
#      rules strict and route client writes through a server when necessary.
#
# 7) Implement write and read operations (server-side)
#    - On every new message or at checkpoints, write a message document to the
#      `messages` subcollection (or append to the array) with a timestamp.
#    - For loading history, query the `messages` subcollection ordered by
#      timestamp and reconstruct the chat_history list to send to the model.
#
# 8) Consider batching and cost control
#    - Firestore charges for reads/writes. Batch frequently or compress older
#      messages into archival storage if you keep long histories.
#    - Use indexes and efficient queries to minimize read costs when loading
#      conversation context.
#
# 9) Backups and exports
#    - Consider scheduled exports or backups of Firestore data to Cloud
#      Storage to retain history and allow recovery.
#
# 10) Monitor and test
#    - Monitor Firestore usage, set alerts for cost spikes, and test security
#      rules and edge cases (concurrent writes, malformed data).
#


# Firebase Project Id is = gpt5py



# Conversation history stored as a list of Message objects
chat_history: list = []

# System prompt that defines assistant behavior. Important: do NOT leave a
# trailing comma here — that would create a tuple instead of a SystemMessage.
system_message = SystemMessage(content="You are a helpful assistant.")
chat_history.append(system_message)

# Initialize the LLM client. Adjust model_name and temperature as needed.
model = ChatOpenAI(model_name="gpt-4o", temperature=0)


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
        result = model.invoke(chat_history)

        # Extract response content and append to history
        response = getattr(result, "content", str(result))
        chat_history.append(AIMessage(content=response))

        # Print the assistant's reply and the updated history
        print(f"AI: {response}\n")
        print_history(chat_history)


if __name__ == "__main__":
    main_loop()