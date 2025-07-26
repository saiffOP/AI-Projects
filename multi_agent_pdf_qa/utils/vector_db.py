import os
import time
import json
import numpy as np
from uuid import uuid4
from pinecone import Pinecone, ServerlessSpec
from openai import AzureOpenAI

from dotenv import load_dotenv


load_dotenv()

PINECONE_APIKEY = os.getenv("PINECONE_APIKEY")
AZURE_APIKEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

# Initialize Azure OpenAI client
azure_client = AzureOpenAI(
    api_key=AZURE_APIKEY,
    api_version="2024-05-01-preview",
    azure_endpoint=AZURE_ENDPOINT
)

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_APIKEY)

# Vector DB index
index_name = "pdf-assistant"
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

pinecone_index = pc.Index(index_name)

def generate_embedding(text: str) -> np.ndarray | None:
    try:
        response = azure_client.embeddings.create(
            input=[text],
            model="embedding_model"
        )
        embedding = json.loads(response.model_dump_json(indent=2))["data"][0]["embedding"]
        return np.array(embedding).reshape(1, 1536)
    except Exception as e:
        print(f"❌ Error generating embedding: {e}")
        return None

def upsert_texts(texts: list[str], metadata_list: list[dict] = None):
    vectors = []
    for i, text in enumerate(texts):
        embedding = generate_embedding(text)
        if embedding is not None:
            metadata = metadata_list[i] if metadata_list and i < len(metadata_list) else {"text": text}
            vectors.append((str(uuid4()), embedding.tolist()[0], metadata))

    if vectors:
        pinecone_index.upsert(vectors)
        print(f"✅ Upserted {len(vectors)} vectors into Pinecone.")

def query_similar_texts(query_text: str, top_k: int = 5) -> list[dict]:
    query_embedding = generate_embedding(query_text)
    if query_embedding is None:
        print("❌ Failed to generate embedding for query.")
        return []

    results = pinecone_index.query(
        vector=query_embedding.tolist()[0],
        top_k=top_k,
        include_metadata=True
    )

    print("\n🔍 Top Matches:")
    for match in results["matches"]:
        score = match.get("score", "N/A")
        snippet = match.get("metadata", {}).get("text", "")
        print(f"Score: {score:.4f} → {snippet[:100]}...")

    return results["matches"]

def delete_uploaded_vectors():
    try:
        # Only proceed if the index exists
        if index_name in [i["name"] for i in pc.list_indexes()]:
            pinecone_index.delete(delete_all=True)
            print("🧹 Cleared all vectors from Pinecone.")
        else:
            print("⚠️ Pinecone index does not exist or was already deleted.")
    except Exception as e:
        print(f"❌ Error while clearing Pinecone index: {e}")





