# populate.py: Script to populate aura-soma index with chunks.json (run only if index needs recreation)

import os
import json
import logging
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables (securely manage these)
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    logger.error("OPENAI_API_KEY not found.")
    raise ValueError("Set OPENAI_API_KEY in your environment.")

pinecone_api_key = os.environ.get("PINECONE_API_KEY")
if not pinecone_api_key:
    logger.error("PINECONE_API_KEY not found.")
    raise ValueError("Set PINECONE_API_KEY in your environment.")

index_name = "aura-soma"
namespace = "aura-soma-velana"

# Initialize Pinecone with serverless spec
try:
    pc = Pinecone(
        api_key=pinecone_api_key,
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    logger.info("Pinecone client initialized.")
except Exception as e:
    logger.error(f"Pinecone init failed: {e}")
    raise

# Load and validate chunks.json (uses ALL chunks for better demo coverage)
texts = []
metadatas = []
try:
    with open("chunks.json", "r") as f:
        chunks = json.load(f)
        if not chunks or not isinstance(chunks, list):
            raise ValueError("chunks.json is empty or invalid.")
        for chunk in chunks:
            if "content" not in chunk or "metadata" not in chunk:
                raise KeyError("Chunk missing 'content' or 'metadata'.")
            texts.append(chunk["content"])
            metadatas.append(chunk["metadata"])
    logger.info(f"Loaded {len(chunks)} chunks (using ALL for comprehensive demo).")
except Exception as e:
    logger.error(f"chunks.json error: {e}")
    raise

# Optional: Filter chunks for demo (e.g., only FAQs and products; uncomment if needed)
# texts = [t for i, t in enumerate(texts) if 'q' in metadatas[i].get('chunk_id', '') or 'products' in metadatas[i].get('topic', '')]
# metadatas = [m for m in metadatas if 'q' in m.get('chunk_id', '') or 'products' in m.get('topic', '')]
# logger.info(f"Filtered to {len(texts)} chunks for demo focus.")

# Initialize embeddings and vector store
try:
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        pinecone_api_key=pinecone_api_key,
        namespace=namespace
    )
    logger.info(f"Vector store ready for {index_name}/{namespace}.")
except Exception as e:
    logger.error(f"Vector store init failed: {e}")
    raise

# Populate (upsert) all vectors
try:
    vectorstore.add_texts(texts=texts, metadatas=metadatas)
    logger.info(f"Populated {len(texts)} vectors successfully.")
    print("Index populated! Ready for demo questions like 'What is Aura-Soma?'.")
except Exception as e:
    logger.error(f"Population failed: {e}")
    raise