try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # dotenv not installed or .env not present — continue without loading env file
    pass

# Try to import the real ChatOpenAI. If the package isn't installed, provide a
# lightweight mock so this script can run for testing without network calls.
try:
    from langchain_openai import ChatOpenAI
    REAL_CHAT = True
except ImportError:
    REAL_CHAT = False

    class ChatOpenAI:
        """Minimal mock ChatOpenAI with the same initializer signature used here.

        The `invoke` method returns an object with a `.content` attribute so the
        rest of the script can remain unchanged for testing.
        """

        def __init__(self, model_name: str = "gpt-4o", temperature: float = 0):
            self.model_name = model_name
            self.temperature = temperature

        def invoke(self, prompt_or_history):
            class Resp:
                def __init__(self, text):
                    self.content = text

            # Return a deterministic mock response that includes the model name.
            return Resp(f"[mock response from {self.model_name}] Hello! I am fine.")

# List of available model names (canonical strings)
models = [
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-4o-mini",
    "gemini-pro",
    "gemini-1.5-pro",
    "gemenai-1.5-flash",
]

# Build a normalized name map once for fast, case-insensitive lookup.
normalized = {m.lower(): m for m in models}

def resolve_model(requested: str, default: str = "gpt-4o") -> str:
    """Resolve a requested model name to a canonical model string.
    - Normalizes the requested name and looks up in the `normalized` map.
    - Returns the default model if not found.
    """
    if not requested:
        return default
    key = requested.lower().strip()
    return normalized.get(key, default)


model_name = resolve_model("gpt-4o-mini")
model = ChatOpenAI(model_name=model_name, temperature=0)
response = model.invoke("Hello, how are you?")
print(response.content)

# Uncomment and test other models as needed.
# model_name = resolve_model("gemenai-1.5-flash")
# model = ChatOpenAI(model_name=model_name, temperature=0)
# response = model.invoke("Hello, how are you?")
# print(response.content)

