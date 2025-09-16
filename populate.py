import os
import json
import logging
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables (set in .env or system)
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "your-openai-api-key")
api_key = os.environ.get("PINECONE_API_KEY", "your-pinecone-api-key")
index_name = "aura-soma"  # Confirmed Pinecone index name
namespace = "aura-soma-velana"  # Matches Colab config

# Load text chunks
try:
    with open("chunks.json", "r") as f:
        chunks = json.load(f)
        texts = [chunk["content"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
    logger.info(f"Loaded {len(chunks)} chunks from chunks.json")
except FileNotFoundError:
    logger.error("chunks.json not found.")
    raise
except json.JSONDecodeError:
    logger.error("Invalid JSON in chunks.json.")
    raise
except KeyError as e:
    logger.error(f"Missing required field in chunks.json: {e}")
    raise

# Initialize Pinecone
pc = Pinecone(api_key=api_key)
indexes = pc.list_indexes().get("indexes", [])

# Check if index exists and validate configuration
expected_dimension = 1024  # Matches Colab config
expected_metric = "cosine"
try:
    if index_name not in [idx["name"] for idx in indexes]:
        logger.warning(f"Index '{index_name}' does not exist. Creating new index.")
        pc.create_index(
            name=index_name,
            dimension=expected_dimension,
            metric=expected_metric,
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        logger.info(f"Created index '{index_name}' in us-east-1.")
    else:
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        actual_dimension = stats.get("dimension")
        actual_metric = stats.get("metric", "unknown")
        vector_count = stats.get("namespaces", {}).get(namespace, {}).get("vector_count", 0)
        if actual_dimension != expected_dimension or actual_metric != expected_metric:
            logger.error(
                f"Index '{index_name}' has dimension={actual_dimension}, metric={actual_metric}, "
                f"but expected dimension={expected_dimension}, metric={expected_metric}."
            )
            raise ValueError("Index configuration mismatch.")
        logger.info(f"Index '{index_name}' exists with {vector_count} vectors in namespace '{namespace}'.")
except Exception as e:
    logger.error(f"Error accessing or creating index: {e}")
    raise

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1024)

# Prepare data with chunk_id as vector ID
vector_data = []
for i, (text, metadata) in enumerate(zip(texts, metadatas)):
    if "chunk_id" not in metadata:
        logger.warning(f"Chunk {i} missing chunk_id; using index {i} as fallback.")
        metadata["chunk_id"] = str(i)
    vector_data.append({
        "id": metadata["chunk_id"],
        "text": text,
        "metadata": metadata
    })

# Get existing IDs to avoid duplicates
index = pc.Index(index_name)
existing_ids = set()
try:
    batch_size = 100
    for i in range(0, len(vector_data), batch_size):
        batch_ids = [vd["id"] for vd in vector_data[i:i + batch_size]]
        fetch_response = index.fetch(ids=batch_ids, namespace=namespace)
        existing_ids.update(fetch_response.get("vectors", {}).keys())
    logger.info(f"Found {len(existing_ids)} existing vectors in namespace '{namespace}'.")
except Exception as e:
    logger.error(f"Error fetching existing vectors: {e}")
    raise

# Filter to new vectors only
new_vector_data = [vd for vd in vector_data if vd["id"] not in existing_ids]
new_texts = [vd["text"] for vd in new_vector_data]
new_metadatas = [vd["metadata"] for vd in new_vector_data]

if not new_texts:
    logger.info("No new chunks to add; all IDs already exist in the index.")
    print(f"Index '{index_name}' (namespace '{namespace}') already up-to-date!")
else:
    logger.info(f"Adding {len(new_texts)} new chunks to the index.")
    batch_size = 100  # Adjust for API limits
    for i in range(0, len(new_texts), batch_size):
        try:
            vector_store = PineconeVectorStore.from_texts(
                texts=new_texts[i:i + batch_size],
                embedding=embeddings,
                index_name=index_name,
                metadatas=new_metadatas[i:i + batch_size],
                ids=[vd["id"] for vd in new_vector_data[i:i + batch_size]],
                namespace=namespace
            )
            logger.info(f"Upserted batch {i // batch_size + 1} of {(len(new_texts) + batch_size - 1) // batch_size}")
        except Exception as e:
            logger.error(f"Error in batch {i // batch_size + 1}: {e}")
            raise

    print(f"Index '{index_name}' (namespace '{namespace}') populated successfully with {len(new_texts)} new vectors!")

# Test query to verify vector store
try:
    vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings, namespace=namespace)
    results = vector_store.similarity_search("What is Aura-Soma?", k=5)
    for i, res in enumerate(results):
        logger.info(f"Query result {i+1}: {res.page_content} (metadata: {res.metadata})")
except Exception as e:
    logger.error(f"Error querying vector store: {e}")
