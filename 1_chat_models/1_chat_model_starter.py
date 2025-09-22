from langchain_openai  import ChatOpenAI
from dotenv import load_dotenv 
load_dotenv()

llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
response = llm.invoke("Hello, how are you?")
print(response.content)