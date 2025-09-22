from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables (e.g., OPENAI_API_KEY)
load_dotenv()

# Initialize model
llm = ChatOpenAI(model="gpt-4o")

# Example 2: Using message roles for system and human messages
messages = [
    ("system",
     "You are a helpful assistant that drafts professional emails for job seekers with the  {tone} email to {company} for the position {position}, mentioning skills {skills} "),
    ("human", "Write a {tone} email to {company}, {position}, {skills}"),
]

prompt_template = ChatPromptTemplate.from_messages(messages)

prompt = prompt_template.invoke({
    "tone": "professional",
    "company": "Acme Corp",
    "position": "interest in the Software Engineer role",
    "skills": "Python, JavaScript, and cloud technologies"
})


# Call the LLM
result = llm.invoke(prompt)
print(result.content)

# Note: with the template prompt you can only use one complete human message at a time.
