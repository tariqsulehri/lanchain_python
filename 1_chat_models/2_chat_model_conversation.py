from langchain_core.messages  import AIMessage, HumanMessage, SystemMessage
from langchain_openai  import ChatOpenAI
from dotenv import load_dotenv 
load_dotenv()


messages = [
    SystemMessage(content="You are as a helpful social media content strategist."),
    HumanMessage(content="Write a tweet about the benefits of using LangChain for building AI applications")
]

llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
response = llm.invoke(messages)
print(response.content)

