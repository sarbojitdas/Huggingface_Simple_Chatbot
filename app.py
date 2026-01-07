import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

# Load .env
load_dotenv()

# Read Groq key from .env
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found in .env file")
    st.stop()

# Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("user", "{question}")
    ]
)

def generate_response(question, model, temperature, max_tokens):
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY.strip(),  # 🔥 strip is critical
        model=model,
        temperature=temperature,
        max_tokens=max_tokens
    )

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question})

# ---------------- UI ----------------
st.title("Groq Q&A Chatbot")



## Select the OpenAI model
#engine=st.sidebar.selectbox("Select Open AI model",["gpt-4o","gpt-4-turbo","gpt-4"])

model = st.sidebar.selectbox(
    "Select model",
    [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "gemma2-9b-it"
    ]
)


## Adjust response parameter
temperature=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)

max_tokens = st.sidebar.slider( "Max Tokens", min_value=256, max_value=2048, value=1024, step=128)



question = st.text_input("Ask your question")

if question:
    response = generate_response(question, model, temperature, max_tokens)
    st.write(response)


