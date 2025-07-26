from agents import Agent, Runner, function_tool, OpenAIChatCompletionsModel
from openai import AsyncAzureOpenAI
from utils.vector_db import query_similar_texts
import os
from dotenv import load_dotenv


load_dotenv()
# print(f"key: {os.getenv('AZURE_OPENAI_API_KEY')}")
# Step 1: Initialize Azure OpenAI client
azure_client = AsyncAzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-05-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT")
)

# Step 2: Define tools
@function_tool
def rag_tool(query: str) -> str:
    matches = query_similar_texts(query, top_k=5)
    context = "\n".join([m["metadata"].get("text", "") for m in matches])
    return f"Relevant content:\n{context}"



@function_tool
def summarizer_tool(file_name: str) -> str:
    import streamlit as st

    if file_name not in st.session_state.file_chunks:
        return f"❌ File '{file_name}' not found. Please upload it first."

    chunks = st.session_state.file_chunks[file_name]
    context = "\n".join(chunks)

    return f"Context for summary:\n{context}"



# Step 4: Define router agent
router_agent = Agent(
    name="Router Agent",
    instructions="If user input looks like a question, use RAG Tool. If it asks for a summary, use Summarization Tool.",
    model=OpenAIChatCompletionsModel(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        openai_client=azure_client
    ),
    tools=[rag_tool, summarizer_tool],
)