from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables (e.g., OPENAI_API_KEY)
load_dotenv()

# Initialize model
llm = ChatOpenAI(model="gpt-4o")

# Define the template
# Example 1: Simple template with one variable
template = "Write a {tone} email to {company} expressing {position} position, mentioning {skills} as a key strength. Keep it to 4 lines max."

# Use ChatPromptTemplate (not ChatMessagePromptTemplate)
prompt_template = ChatPromptTemplate.from_template(template)

# Fill in variables
prompt = prompt_template.invoke({
    "tone": "professional",
    "company": "Acme Corp",
    "position": "interest in the Software Engineer role",
    "skills": "Python, JavaScript, and cloud technologies"
})

# Call the LLM
result = llm.invoke(prompt)
print(result.content)
