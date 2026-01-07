import streamlit as st
import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --------------------------------------------------
# Load environment variables
# --------------------------------------------------
load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

if not HF_TOKEN:
    st.error("❌ HUGGINGFACE_API_TOKEN not found in environment variables")
    st.stop()

# --------------------------------------------------
# LLM Response Function
# --------------------------------------------------
def generate_response(question, model, temperature, max_tokens):

    # Hugging Face Inference Endpoint (chat-only model)
    endpoint = HuggingFaceEndpoint(
        repo_id=model,
        task="conversational",          # 🔥 REQUIRED for Mistral-Instruct
        temperature=temperature,
        max_new_tokens=max_tokens,
        huggingfacehub_api_token=HF_TOKEN,
    )

    # Wrap endpoint as a Chat model (IMPORTANT)
    llm = ChatHuggingFace(llm=endpoint)

    # Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant."),
        ("human", "{question}")
    ])

    # Chain
    chain = prompt | llm | StrOutputParser()

    return chain.invoke({"question": question})

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
st.set_page_config(page_title="🤗 HuggingFace Q&A Chatbot", page_icon="🤗")

st.title("🤗 HuggingFace Q&A Chatbot")
st.write("Ask questions using **Chat Models** via Hugging Face Inference API")

question = st.text_input("Ask a question")

model = st.sidebar.selectbox(
    "Select Model",
    ["meta-llama/Meta-Llama-3-8B-Instruct","Qwen/Qwen2.5-7B-Instruct","HuggingFaceH4/zephyr-7b-beta"]
)

temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)

max_tokens = st.sidebar.slider("Max Tokens", 100, 1000, 500)

if st.button("Ask"):
    if question.strip():
        with st.spinner("Generating response..."):
            response = generate_response(
                question=question,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
        st.success("Response:")
        st.write(response)
    else:
        st.warning("Please enter a question.")
