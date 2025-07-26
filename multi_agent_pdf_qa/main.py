import os
import streamlit as st
from utils.pdf_utils import extract_text_from_pdf, chunk_text
from utils.vector_db import upsert_texts, generate_embedding, delete_uploaded_vectors
from agent.pdf_agents import router_agent
from agents import Runner
import asyncio
import atexit
from pinecone import Pinecone
from dotenv import load_dotenv


load_dotenv()

# Top-level
pc = Pinecone(api_key=os.getenv("PINECONE_APIKEY"))
pinecone_index = pc.Index("pdf-assistant")


# Initialize session state for chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.set_page_config(page_title="Multi-Agent PDF Assistant", page_icon="🧠")
st.title("📄 Multi-Agent PDF Assistant")

# --- File Upload ---
uploaded_files = st.file_uploader(
    "Upload one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

# --- Text Input ---
user_query = st.text_input("Enter your question or request a summary", placeholder="e.g., Summarize this document or What is the conclusion?")

# --- Submit Button ---
if st.button("Submit") and user_query:
    if not uploaded_files:
        st.warning("Please upload at least one PDF before asking a question.")
    else:
        with st.spinner("🧠 Thinking..."):

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            result = loop.run_until_complete(Runner.run(router_agent, user_query))


            final_output = result.final_output
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            st.session_state.chat_history.append({"role": "agent", "content": final_output})



# --- Display Chat History ---
st.markdown("### 📝 Chat History")
for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        st.markdown(f"**👤 You:** {chat['content']}")
    else:
        st.markdown(f"**🤖 Assistant:** {chat['content']}")

# Optional: Clear Chat Button
if st.button("Clear Chat"):
    st.session_state.chat_history = []


if "file_chunks" not in st.session_state:
    st.session_state.file_chunks = {}

for file in uploaded_files:
    if file.name not in st.session_state.file_chunks:
        text = extract_text_from_pdf(file)
        chunks = chunk_text(text)
        st.session_state.file_chunks[file.name] = chunks


        preview_text = " ".join(chunks[:2])  # Just first 2 chunks for lightweight summary
        embedding = generate_embedding(preview_text)

        if "file_summaries" not in st.session_state:
            st.session_state.file_summaries = {}
        st.session_state.file_summaries[file.name] = embedding


        # ✅ Upsert chunks with metadata so we know which file each chunk came from
        chunks_with_metadata = [
            {"text": chunk, "metadata": {"file_name": file.name}} for chunk in chunks
        ]
        texts = [c["text"] for c in chunks_with_metadata]
        metadata_list = [c["metadata"] for c in chunks_with_metadata]

        upsert_texts(texts, metadata_list)


# Delete Pinecone data when app stops
atexit.register(delete_uploaded_vectors)


