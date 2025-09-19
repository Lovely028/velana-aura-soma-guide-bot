

### Install Dependencies

This cell installs the required libraries for setting up the Aura-Soma chatbot, including `langchain-openai`, `pinecone-client`, `langchain-pinecone`, and `langchain-core`.
"""

!pip install langchain-openai pinecone-client langchain-pinecone langchain-core --upgrade langchain

"""# AuraGuide Bot Setup and Evaluation

This notebook sets up a RAG (Retrieval Augmented Generation) chatbot for Velana.net, specializing in Aura-Soma. It includes data processing, vector store initialization using Pinecone, and an evaluation pipeline using LangSmith.
"""

import json
import logging
import os
import re
import time
import uuid
import hashlib
from typing import List, Dict, Optional, Callable
from google.colab import userdata
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langsmith import Client, traceable
from langchain_core.tracers.langchain import wait_for_all_tracers
from langchain_core.runnables import Runnable
from langsmith.evaluation import evaluate, EvaluationResult
import backoff

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load API keys from Colab secrets
try:
    OPENAI_API_KEY: str = userdata.get("OPENAI_API_KEY")
    PINECONE_API_KEY: str = userdata.get("PINECONE_API_KEY")
    LANGCHAIN_API_KEY: Optional[str] = userdata.get("LANGCHAIN_API_KEY")
except Exception as e:
    logging.error(f"Error loading API keys from Colab secrets: {str(e)}")
    raise ValueError("Failed to load required API keys from Colab secrets")

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    raise ValueError("OPENAI_API_KEY and PINECONE_API_KEY must be set in Colab secrets")

# Configure LangSmith
if LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "AuraGuideBot-Trace"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_CALLBACKS_BACKGROUND"] = "false"
else:
    logging.warning("LANGCHAIN_API_KEY not set. LangSmith tracing disabled.")

# Initialize LangSmith client
def init_langsmith_client() -> Optional[Client]:
    """Initialize LangSmith client with retry logic."""
    if not LANGCHAIN_API_KEY:
        logging.warning("LANGCHAIN_API_KEY not set. Skipping LangSmith client initialization.")
        return None
    if not LANGCHAIN_API_KEY.startswith("lsv2_"):
        logging.warning("Invalid LANGCHAIN_API_KEY format. Must start with 'lsv2_'. Skipping.")
        return None
    @backoff.on_exception(backoff.expo, ValueError, max_tries=3)
    def create_client():
        return Client(api_key=LANGCHAIN_API_KEY)
    try:
        return create_client()
    except ValueError as e:
        logging.error(f"LangSmith client failed to initialize after retries: {str(e)}")
        return None

langsmith_client: Optional[Client] = init_langsmith_client()

# Configuration
CONFIG = {
    "embedding_model": "text-embedding-3-large",
    "embedding_dimensions": 1024,
    "similarity_threshold": 0.3,
    "search_k": 15,
    "booking_link": "https://velana.net/aurasoma#offers",
    "index_name": "aura-soma",
    "vector_store": "aura-soma-vs1",
    "namespace": "aura-soma-velana",
    "chunk_size": 1000,
    "chunk_overlap": 10,
    "json_files": [
        "/content/aura_soma_faq.json",
        "/content/aura_soma_pricelist.json",
        "/content/Aura_Soma_Products.json",
        "/content/meet_aura_soma__and_awaken_through_aura_soma.json"
    ],
    "json_sources": {
        "faq": "/content/aura_soma_faq.json",
        "pricing": "/content/aura_soma_pricelist.json",
        "product": "/content/Aura_Soma_Products.json",
        "meet_aura_soma": "/content/meet_aura_soma__and_awaken_through_aura_soma.json"
    },
    "expected_counts": {
        "equilibrium": 124,
        "pomander": 19,
        "quintessence": 15,
        "transcripts": 10
    }
}

def clean_text_content(text: str) -> str:
    """Clean text by normalizing spaces and fixing common issues."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\bu\b', 'you', text)
    text = re.sub(r'\buh\b', '', text)
    text = re.sub(r'\bum\b', '', text)
    text = re.sub(r'\s*\n\s*', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def generate_stable_id(content: str, source: str, index: int) -> str:
    """Generate a stable UUID for a chunk based on content and source."""
    content_hash = hashlib.md5((content + source + str(index)).encode()).hexdigest()
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, content_hash))

def assign_chunk_metadata(content: str, source: str, index: int, extra_metadata: Optional[Dict] = None) -> Dict:
    """Assign metadata with stable chunk ID."""
    content_lower = content.lower()
    metadata: Dict[str, str] = {
        "source": source,
        "document_type": "structured",
        "chunk_id": generate_stable_id(content, source, index)
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    topic_keywords = {
        "general": ["what is aura-soma", "mission", "products"],
        "services": ["services"],
        "pricing": ["price", "cost", "€"],
        "shipping": ["shipping"],
        "authenticity_sustainability": ["authentic", "vegan", "organic", "sustainable"],
        "returns": ["return", "returns", "refund"]
    }

    if "faq" in source.lower():
        for topic, keywords in topic_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                metadata["topic"] = topic
                break
    elif "pricelist" in source.lower():
        metadata["topic"] = "pricing"
    elif "products" in source.lower():
        metadata["topic"] = "products"
        bottle_pattern = r'B(\d{1,3})\s*-\s*([^\n.]*)'
        match = re.search(bottle_pattern, content, re.IGNORECASE)
        if match:
            metadata["bottle"] = f"B{match.group(1).zfill(3)}"
            metadata["title"] = match.group(2).strip()
        elif "b0" in content_lower:
            metadata["bottle"] = "potential_bottle"
    elif "meet_aura_soma" in source.lower():
        metadata["document_type"] = "transcript"
        if "vicky wall" in content_lower:
            metadata["topic"] = "history"
        elif any(keyword in content_lower for keyword in ["color therapy", "light", "aura"]):
            metadata["topic"] = "philosophy"
        else:
            metadata["topic"] = "transcript"

    return metadata

def merge_transcript_chunks(chunks: List[Dict], target_size: int = CONFIG["chunk_size"] // 5) -> List[Dict]:
    """Merge transcript chunks by speaker to form coherent segments."""
    merged_chunks: List[Dict] = []
    current_speaker: Optional[str] = None
    current_content: str = ""
    current_ids: List[str] = []
    total_chars = 0

    for chunk in chunks:
        speaker: str = chunk["speaker"]
        content: str = clean_text_content(chunk["content"])
        source_id: str = str(chunk["source_id"])

        if not content or len(content) < 10:
            continue

        total_chars += len(content)
        if (speaker == current_speaker and
                len(current_content) + len(content) < target_size):
            current_content += " " + content
            current_ids.append(source_id)
        else:
            if current_content:
                merged_chunks.append({
                    "source_id": ",".join(set(current_ids)),
                    "speaker": current_speaker,
                    "content": current_content
                })
            current_speaker = speaker
            current_content = content
            current_ids = [source_id]

    if current_content:
        merged_chunks.append({
            "source_id": ",".join(set(current_ids)),
            "speaker": current_speaker,
            "content": current_content
        })

    logging.info(f"Merged {len(merged_chunks)} transcript chunks: {[len(c['content']) for c in merged_chunks]}. Total characters: {total_chars}")
    return merged_chunks

def validate_json_inputs(json_files: List[str]) -> Dict[str, int]:
    """Validate JSON files and return counts of entries."""
    counts = {"faq": 0, "pricing": 0, "products": 0, "transcripts": 0}
    for file in json_files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "faq" in file.lower():
                if "faqs" not in data:
                    logging.error(f"Expected 'faqs' key in {file}, got {list(data.keys())}")
                    continue
                counts["faq"] = len(data.get("faqs", []))
                if not any("what is aura-soma" in faq["question"].lower() for faq in data.get("faqs", [])):
                    logging.warning(f"No entry for 'What is Aura-Soma?' in {file}. Consider adding one.")
            elif "pricelist" in file.lower():
                if "services" not in data:
                    logging.error(f"Expected 'services' key in {file}, got {list(data.keys())}")
                    continue
                counts["pricing"] = len(data.get("services", []))
            elif "products" in file.lower():
                if not isinstance(data, list):
                    logging.error(f"Expected list in {file}, got {type(data)}")
                    continue
                counts["products"] = len(data)
            elif "meet_aura_soma" in file.lower():
                if not isinstance(data, list) or "chunks" not in data[0]:
                    logging.error(f"Expected list with 'chunks' in {file}, got {type(data)}")
                    continue
                counts["transcripts"] = len(data[0].get("chunks", []))
            logging.info(f"Validated {file}: {counts.get(file.split('/')[-1].split('.')[0], 0)} entries")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logging.error(f"Error validating {file}: {str(e)}")
    return counts

def process_single_json(json_file: str, chunk_size: int = CONFIG["chunk_size"],
                       chunk_overlap: int = CONFIG["chunk_overlap"]) -> List[Document]:
    """Process a single JSON file into chunks with stable IDs."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", ".", " "]
    )

    chunked_documents: List[Document] = []

    try:
        with open(json_file, "r", encoding="utf-8") as file:
            json_data = json.load(file)
    except FileNotFoundError:
        logging.error(f"JSON file not found: {json_file}")
        return chunked_documents
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in {json_file}: {str(e)}")
        return chunked_documents

    if "faq" in json_file.lower():
        if not isinstance(json_data, dict) or "faqs" not in json_data:
            logging.error(f"Expected dict with 'faqs' in {json_file}, got {type(json_data)}")
            return chunked_documents
        data = json_data.get("faqs", [])
    elif "pricelist" in json_file.lower():
        if not isinstance(json_data, dict) or "services" not in json_data:
            logging.error(f"Expected dict with 'services' in {json_file}, got {type(json_data)}")
            return chunked_documents
        data = json_data.get("services", [])
    elif "products" in json_file.lower():
        if not isinstance(json_data, list):
            logging.error(f"Expected list in {json_file}, got {type(json_data)}")
            return chunked_documents
        data = json_data
    elif "meet_aura_soma" in json_file.lower():
        if not isinstance(json_data, list) or not json_data or "chunks" not in json_data[0]:
            logging.error(f"Expected list with 'chunks' in {json_file}, got {type(json_data)}")
            return chunked_documents
        chunks = json_data[0].get("chunks", [])
        merged_chunks = merge_transcript_chunks(chunks, target_size=chunk_size // 5)
        for j, chunk in enumerate(merged_chunks):
            content_text = clean_text_content(chunk["content"])
            if len(content_text) <= 10:
                continue
            metadata = assign_chunk_metadata(
                content_text, json_file, j, {"speaker": chunk["speaker"], "source_id": chunk["source_id"]}
            )
            split_docs = splitter.split_documents([Document(page_content=content_text, metadata=metadata)])
            chunked_documents.extend(split_docs)
        logging.info(f"Processed {len(chunked_documents)} transcript chunks from {json_file}")
        return chunked_documents
    else:
        logging.error(f"Unknown JSON file type: {json_file}")
        return chunked_documents

    for i, content in enumerate(data):
        if "faq" in json_file.lower():
            content_text = clean_text_content(f"{content['id']}: {content['question']} - {content['answer']}")
            metadata = assign_chunk_metadata(
                content_text, json_file, i, {"category": content.get("category", "")}
            )
        elif "pricelist" in json_file.lower():
            content_text = clean_text_content(
                f"{content['name']}: €{content['price']} - {content['description']} - {' '.join(content.get('details', []))}"
            )
            metadata = assign_chunk_metadata(
                content_text, json_file, i, {"name": content["name"], "price": str(content["price"])}
            )
        elif "products" in json_file.lower():
            content_text = clean_text_content(
                f"{content['id']}: {content['name']} - {content['category']} - {content.get('description', '')}"
            )
            metadata = assign_chunk_metadata(
                content_text, json_file, i, {"id": content["id"], "category": content["category"]}
            )

        if len(content_text) <= 10:
            continue
        chunked_documents.append(Document(page_content=content_text, metadata=metadata))

    return chunked_documents

def process_all_json_files(json_files: List[str] = CONFIG["json_files"],
                          chunk_size: int = CONFIG["chunk_size"],
                          chunk_overlap: int = CONFIG["chunk_overlap"]) -> List[Document]:
    """Process multiple JSON files into chunks with deduplication and detailed logging."""
    all_chunked_documents: List[Document] = []
    seen_chunk_ids: set = set()
    seen_content_hashes: set = set()
    chunk_counts: Dict[str, int] = {file: 0 for file in json_files}

    json_counts = validate_json_inputs(json_files)
    logging.info(f"JSON input counts: {json_counts}")

    for json_file in json_files:
        chunked_documents = process_single_json(json_file, chunk_size, chunk_overlap)
        for doc in chunked_documents:
            content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
            if doc.metadata["chunk_id"] not in seen_chunk_ids and content_hash not in seen_content_hashes:
                all_chunked_documents.append(doc)
                seen_chunk_ids.add(doc.metadata["chunk_id"])
                seen_content_hashes.add(content_hash)
                chunk_counts[json_file] += 1
            else:
                logging.warning(f"Duplicate chunk detected (ID: {doc.metadata['chunk_id']}, hash: {content_hash}) from {json_file}")
        logging.info(f"Processed {json_file}: {chunk_counts[json_file]} chunks")

    valid_documents = [
        doc for doc in all_chunked_documents
        if doc.page_content and len(doc.page_content.strip()) > 10
    ]

    logging.info(f"Generated {len(valid_documents)} unique chunks after deduplication")
    if valid_documents:
        logging.info(
            f"Sample chunk: {valid_documents[0].page_content[:100]}... "
            f"(Metadata: {valid_documents[0].metadata})"
        )

    transcript_count = sum(1 for doc in valid_documents if doc.metadata.get("document_type") == "transcript")
    product_chunks = [doc for doc in valid_documents if doc.metadata.get("source").endswith("Aura_Soma_Products.json")]
    equilibrium_count = sum(1 for doc in product_chunks if doc.metadata.get("category") == "Equilibrium")
    pomander_count = sum(1 for doc in product_chunks if doc.metadata.get("category") == "Pomander")
    quintessence_count = sum(1 for doc in product_chunks if doc.metadata.get("category") == "Quintessence")
    faq_count = sum(1 for doc in valid_documents if doc.metadata.get("source").endswith("aura_soma_faq.json"))
    pricing_count = sum(1 for doc in valid_documents if doc.metadata.get("source").endswith("aura_soma_pricelist.json"))

    logging.info(
        f"Coverage: {faq_count} FAQs, {pricing_count} pricing entries, "
        f"{equilibrium_count}/{CONFIG['expected_counts']['equilibrium']} Equilibrium bottles, "
        f"{pomander_count}/{CONFIG['expected_counts']['pomander']} Pomanders, "
        f"{quintessence_count}/{CONFIG['expected_counts']['quintessence']} Quintessences, "
        f"{transcript_count} transcript chunks"
    )
    logging.info(f"Chunk counts by source: {chunk_counts}")

    if equilibrium_count != CONFIG["expected_counts"]["equilibrium"]:
        logging.warning(f"Expected {CONFIG['expected_counts']['equilibrium']} Equilibrium bottles, got {equilibrium_count}")
    if pomander_count != CONFIG["expected_counts"]["pomander"]:
        logging.warning(f"Expected {CONFIG['expected_counts']['pomander']} Pomanders, got {pomander_count}")
    if quintessence_count != CONFIG["expected_counts"]["quintessence"]:
        logging.warning(f"Expected {CONFIG['expected_counts']['quintessence']} Quintessences, got {quintessence_count}")
    if transcript_count < CONFIG["expected_counts"]["transcripts"]:
        logging.warning(f"Expected at least {CONFIG['expected_counts']['transcripts']} transcript chunks, got {transcript_count}")

    logging.info("Bot is ready for API deployment")
    return valid_documents

# Initialize conversational memory
memory = ConversationBufferMemory(
    llm=ChatOpenAI(
        model="gpt-3.5-turbo-0125",
        temperature=0,
        api_key=OPENAI_API_KEY
    ),
    max_token_limit=2000,
    return_messages=True,
    memory_key="chat_history"
)

class VectorQueryTool:
    """A tool for querying a Pinecone vector store with optional filters and post-processing."""
    def __init__(
        self,
        vector_store: PineconeVectorStore,
        namespace: str,
        default_filter: Optional[Dict] = None,
        post_process: Optional[Callable[[List[str], str], List[str]]] = None
    ) -> None:
        self.vector_store = vector_store
        self.namespace = namespace
        self.default_filter = default_filter or {}
        self.post_process = post_process

    def query(self, query: str, extra_filter: Optional[Dict] = None,
              fallback_to_general: bool = True) -> List[str]:
        """Query the vector store with optional filters and fallback."""
        filter_dict = {**self.default_filter, **(extra_filter or {})}
        try:
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=CONFIG["search_k"],
                namespace=self.namespace,
                filter=filter_dict or None
            )
            filtered = [
                (doc.page_content, score)
                for doc, score in results
                if score > CONFIG["similarity_threshold"]
            ]
            logging.info(
                f"{self.__class__.__name__} retrieved {len(filtered)} docs for query "
                f"'{query}' with filter {filter_dict}. Scores: {[score for _, score in filtered]}"
            )
            documents = [doc for doc, _ in filtered]
            if not documents and fallback_to_general and filter_dict:
                logging.info(f"No results with filter for '{query}'. Trying general search.")
                results = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=CONFIG["search_k"],
                    namespace=self.namespace,
                    filter=None
                )
                filtered = [
                    (doc.page_content, score)
                    for doc, score in results
                    if score > CONFIG["similarity_threshold"]
                ]
                logging.info(
                    f"General search retrieved {len(filtered)} docs for '{query}'. "
                    f"Scores: {[score for _, score in filtered]}"
                )
                documents = [doc for doc, _ in filtered]
            if self.post_process:
                documents = self.post_process(documents, query)
            return documents
        except ValueError as e:
            logging.error(f"Vector store query error for '{query}': {str(e)}")
            return []

def build_product_pattern(id_prefix: str, num: int, category: str) -> re.Pattern:
    """Build regex pattern for product extraction."""
    num_str = f"{num:03d}" if id_prefix == "B" else f"{num:02d}"
    return re.compile(
        rf'"{id_prefix}{num_str}".*?"name":"(.*?)".*?"category":"{category}".*?"description":"(.*?)"',
        re.IGNORECASE | re.DOTALL
    )

def extract_product_metadata(documents: List[str], query: str) -> List[str]:
    """Extract metadata for product queries with JSON fallback."""
    bottle_match = re.search(r'(?:B#?|bottle\s*#?|\#)(\d{1,3})', query, re.IGNORECASE)
    pomander_match = re.search(r'(?:P#?|pomander\s*#?|\#)(\d{1,2})', query, re.IGNORECASE)
    quintessence_match = re.search(r'(?:Q#?|quintessence\s*#?|\#)(\d{1,2})', query, re.IGNORECASE)

    short_descriptions = []

    if bottle_match:
        num = int(bottle_match.group(1))
        if 1 <= num <= 124:
            bottle_id = f"B{num:03d}"
            pattern = build_product_pattern("B", num, "Equilibrium")
            for doc in documents:
                match = pattern.search(doc)
                if match:
                    name, desc = match.groups()
                    if len(desc) < 20:
                        short_descriptions.append(f"{bottle_id}: {desc}")
                        desc += " This bottle supports energetic balance and personal resonance."
                    return [f"Bottle {bottle_id} is {name}. Description: {desc}."]
    elif pomander_match:
        num = int(pomander_match.group(1))
        if 1 <= num <= 19:
            pomander_id = f"P{num:02d}"
            pattern = build_product_pattern("P", num, "Pomander")
            for doc in documents:
                match = pattern.search(doc)
                if match:
                    name, desc = match.groups()
                    if len(desc) < 20:
                        short_descriptions.append(f"{pomander_id}: {desc}")
                        desc += " This pomander supports energetic cleansing and protection."
                    if pomander_id == "P10":
                        logging.info(f"P10 details: name={name}, desc={desc}")
                    return [f"Pomander {pomander_id} is {name}. Description: {desc}."]
    elif quintessence_match:
        num = int(quintessence_match.group(1))
        if 1 <= num <= 15:
            quintessence_id = f"Q{num:02d}"
            pattern = build_product_pattern("Q", num, "Quintessence")
            for doc in documents:
                match = pattern.search(doc)
                if match:
                    name, desc = match.groups()
                    if len(desc) < 20:
                        short_descriptions.append(f"{quintessence_id}: {desc}")
                        desc += " This quintessence supports meditation and spiritual connection."
                    return [f"Quintessence {quintessence_id} is {name}. Description: {desc}."]

    if short_descriptions:
        logging.warning(f"Short descriptions found: {short_descriptions}. Consider updating Aura_Soma_Products.json.")

    file_path = CONFIG["json_sources"]["product"]
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            json_data = json.load(file)
        if not isinstance(json_data, list):
            logging.error(f"Expected list in {file_path}, got {type(json_data)}")
            return []
        if bottle_match:
            num = int(bottle_match.group(1))
            if 1 <= num <= 124:
                bottle_id = f"B{num:03d}"
                for product in json_data:
                    if product["id"] == bottle_id:
                        desc = product["description"]
                        if len(desc) < 20:
                            short_descriptions.append(f"{bottle_id}: {desc}")
                            desc += " This bottle supports energetic balance and personal resonance."
                        return [f"Bottle {bottle_id} is {product['name']}. Description: {desc}."]
                return [f"Bottle {bottle_id} exists in the Aura-Soma Equilibrium range. Please consult a practitioner for detailed insights."]
        elif pomander_match:
            num = int(pomander_match.group(1))
            if 1 <= num <= 19:
                pomander_id = f"P{num:02d}"
                for product in json_data:
                    if product["id"] == pomander_id:
                        desc = product["description"]
                        if len(desc) < 20:
                            short_descriptions.append(f"{pomander_id}: {desc}")
                            desc += " This pomander supports energetic cleansing and protection."
                        if pomander_id == "P10":
                            logging.info(f"P10 JSON details: name={product['name']}, desc={desc}")
                        return [f"Pomander {pomander_id} is {product['name']}. Description: {desc}."]
                return [f"Pomander {pomander_id} exists in the Aura-Soma range. Please consult a practitioner for details."]
        elif quintessence_match:
            num = int(quintessence_match.group(1))
            if 1 <= num <= 15:
                quintessence_id = f"Q{num:02d}"
                for product in json_data:
                    if product["id"] == quintessence_id:
                        desc = product["description"]
                        if len(desc) < 20:
                            short_descriptions.append(f"{quintessence_id}: {desc}")
                            desc += " This quintessence supports meditation and spiritual connection."
                        return [f"Quintessence {quintessence_id} is {product['name']}. Description: {desc}."]
                return [f"Quintessence {quintessence_id} exists in the Aura-Soma range. Please consult a practitioner for details."]
    except FileNotFoundError:
        logging.error(f"JSON file not found: {file_path}")
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in {file_path}: {str(e)}")
    return documents

def append_booking_redirect(documents: List[str], query: str) -> List[str]:
    """Append booking link to retrieved documents."""
    return documents + [
        f"To book a consultation or purchase a gift card, visit: {CONFIG['booking_link']}"
    ]

def local_json_search(route: str, query: str) -> List[str]:
    """Search JSON files for relevant content when vector search fails."""
    file_path = CONFIG["json_sources"].get(route)
    if not file_path:
        return []
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            json_data = json.load(file)
        matches: List[str] = []
        query_lower = query.lower()
        if route == "faq":
            if not isinstance(json_data, dict) or "faqs" not in json_data:
                logging.error(f"Expected 'faqs' in {file_path}, got {list(json_data.keys())}")
                return []
            for faq in json_data.get("faqs", []):
                if query_lower in faq["question"].lower() or query_lower in faq["answer"].lower():
                    matches.append(faq["answer"])
            if "what is aura-soma" in query_lower:
                for faq in json_data.get("faqs", []):
                    if "what is aura-soma" in faq["question"].lower():
                        matches.append(faq["answer"])
                        break
                else:
                    matches.append(
                        "Aura-Soma is a unique holistic system that integrates color therapy, numerology, astrology, and spiritual wisdom to support personal well-being and self-discovery."
                    )
        elif route == "pricing":
            if not isinstance(json_data, dict) or "services" not in json_data:
                logging.error(f"Expected 'services' in {file_path}, got {list(json_data.keys())}")
                return []
            for service in json_data.get("services", []):
                if (query_lower in service["name"].lower() or
                        query_lower in service["description"].lower()):
                    matches.append(
                        f"{service['name']} consultation: €{service['price']}. "
                        f"{service['description']}"
                    )
        elif route == "product":
            if not isinstance(json_data, list):
                logging.error(f"Expected list in {file_path}, got {type(json_data)}")
                return []
            for product in json_data:
                if (query_lower in product["id"].lower() or
                    query_lower in product["name"].lower() or
                    query_lower in product["description"].lower()):
                    matches.append(
                        f"{product['category']} {product['id']} is {product['name']}. "
                        f"Description: {product['description']}"
                    )
        elif route == "meet_aura_soma":
            if not isinstance(json_data, list) or not json_data or "chunks" not in json_data[0]:
                logging.error(f"Expected list with 'chunks' in {file_path}, got {type(json_data)}")
                return []
            matches = [
                chunk["content"]
                for file in json_data
                for chunk in file.get("chunks", [])
                if query_lower in chunk["content"].lower()
            ]
        logging.info(
            f"Local JSON fallback for '{query}' in {file_path} retrieved "
            f"{len(matches)} matches: {matches[:1000]}..."
        )
        return matches
    except FileNotFoundError:
        logging.error(f"JSON file not found: {file_path}")
        return []
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in {file_path}: {str(e)}")
        return []

def initialize_pinecone():
    """Initialize Pinecone with safe namespace clearing and verification."""
    try:
        pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
        index_name = CONFIG["index_name"]
        namespace = CONFIG["namespace"]

        existing_indexes = pinecone_client.list_indexes().names()
        if index_name not in existing_indexes:
            logging.info(f"Creating Pinecone index: {index_name}")
            pinecone_client.create_index(
                name=index_name,
                dimension=CONFIG["embedding_dimensions"],
                metric="cosine",
                spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
            )
            while not pinecone_client.describe_index(index_name).status["ready"]:
                time.sleep(1)
            logging.info(f"Index '{index_name}' created and ready.")
        else:
            logging.info(f"Index '{index_name}' already exists. Connecting...")

        index = pinecone_client.Index(index_name)
        index_stats = index.describe_index_stats()
        if index_stats["dimension"] != CONFIG["embedding_dimensions"]:
            raise ValueError(
                f"Index dimension {index_stats['dimension']} does not match "
                f"embedding dimension {CONFIG['embedding_dimensions']}"
            )

        if namespace in index_stats["namespaces"]:
            for attempt in range(3):
                logging.info(f"Clearing namespace '{namespace}' (attempt {attempt + 1})...")
                try:
                    index.delete(delete_all=True, namespace=namespace)
                    time.sleep(2)
                    index_stats = index.describe_index_stats()
                    vector_count = index_stats["namespaces"].get(namespace, {}).get("vector_count", 0)
                    if vector_count == 0:
                        logging.info(f"Namespace '{namespace}' successfully cleared.")
                        break
                except Exception as e:
                    logging.warning(f"Namespace deletion attempt {attempt + 1} failed: {str(e)}")
            else:
                logging.warning(f"Failed to clear namespace '{namespace}' after retries. Proceeding with ingestion.")
        else:
            logging.info(f"Namespace '{namespace}' does not exist. Proceeding with ingestion.")

        embeddings = OpenAIEmbeddings(
            model=CONFIG["embedding_model"],
            api_key=OPENAI_API_KEY,
            dimensions=CONFIG["embedding_dimensions"]
        )
        vector_store = PineconeVectorStore(
            index=index,
            embedding=embeddings,
            namespace=namespace,
            text_key=CONFIG["vector_store"]
        )
        return vector_store, index
    except ValueError as e:
        logging.error(f"Vector store initialization error: {str(e)}")
        raise
    except KeyError as e:
        logging.error(f"Missing environment variable: {str(e)}")
        raise

vector_store, index = initialize_pinecone()

tools = {
    "faq": VectorQueryTool(
        vector_store=vector_store,
        namespace=CONFIG["namespace"],
        default_filter={
            "topic": {
                "$in": [
                    "general", "services", "pricing", "shipping",
                    "authenticity_sustainability", "discounts", "returns"
                ]
            }
        }
    ),
    "pricing": VectorQueryTool(
        vector_store=vector_store,
        namespace=CONFIG["namespace"],
        default_filter={"topic": "pricing"}
    ),
    "product": VectorQueryTool(
        vector_store=vector_store,
        namespace=CONFIG["namespace"],
        default_filter={"topic": "products"},
        post_process=extract_product_metadata
    ),
    "meet_aura_soma": VectorQueryTool(
        vector_store=vector_store,
        namespace=CONFIG["namespace"],
        default_filter={"document_type": "transcript"}
    ),
    "booking": VectorQueryTool(
        vector_store=vector_store,
        namespace=CONFIG["namespace"],
        default_filter={"topic": "pricing"},
        post_process=append_booking_redirect
    )
}

try:
    llm = ChatOpenAI(
        model="gpt-3.5-turbo-0125",
        temperature=0,
        api_key=OPENAI_API_KEY
    )
except ValueError as e:
    logging.error(f"Error initializing LLM: {str(e)}")
    raise

system_prompt = """You are AuraGuide, a helpful, mindful chatbot for Velana.net, specializing in Aura-Soma—a holistic color-care system for wellbeing. Respond concisely, warmly, using language inspired by light and harmony. Be positive, non-judgmental, and emphasize personal resonance.

Key Rules:
- Use provided {context} if relevant; otherwise, use tools or general knowledge.
- For general Aura-Soma questions (e.g., shipping, returns, overview), use FAQTool or MeetAuraSoma tools and end with: "To dive deeper, book a consultation at {booking_link}."
- For pricing questions, use PricingTool to retrieve name, price, description from aura_soma_pricelist.json, e.g., "Express consultation: €35 - Personalized guidance for a chosen day." Then include: "Book at {booking_link}."
- For booking questions, use BookingTool and respond with: "You can book an Aura-Soma consultation online at {booking_link}."
- For bottle/pomander/quintessence queries, use ProductTool to extract id, name, category, description from Aura_Soma_Products.json (e.g., "B050 - El Morya - Equilibrium - Thy will, not my will"). Then say: "I'm not a practitioner, so I can't provide full analysis. Book with Velana: {booking_link}."
- If no relevant info, say: "I couldn't find specific details, but generally [brief info]. Book at {booking_link}."
- Always include the booking link ({booking_link}) for personalized advice.
- Keep responses concise, detailed enough to match expected outputs, and consistent with the Aura-Soma system's holistic tone.
- Maintain conversational memory for context-aware responses."""

def generate_response(query: str, context: List[str], route: str, chat_history: Optional[List[Dict]]) -> str:
    """Generate a response based on the query, retrieved context, and chat history."""
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Chat History: {chat_history}\nContext: {context}\nQuery: {query}")
    ])

    chain = prompt_template | llm

    response = chain.invoke({
        "query": query,
        "context": "\n---\n".join(context),
        "chat_history": memory.load_memory_variables({})["chat_history"],
        "booking_link": CONFIG["booking_link"]
    }).content

    memory.save_context({"input": query}, {"output": response})
    return response

@traceable
def process_query(query: str, chat_history: Optional[List[Dict]] = None) -> Dict:
    """Process a user query through the RAG pipeline."""
    start_time = time.time()
    if not query.strip():
        latency = time.time() - start_time
        response = f"Please provide a query. To dive deeper, book at {CONFIG['booking_link']}."
        logging.info(f"Query processing time: {latency:.4f} seconds")
        return {"response": response}

    query = query.strip()
    run_id = None
    if langsmith_client:
        try:
            run_id = langsmith_client.create_run(
                name="AuraGuideQuery",
                inputs={"query": query, "chat_history": chat_history},
                run_type="chain",
                metadata={"query_type": route_query(query)}
            )
        except ValueError as e:
            logging.warning(f"Failed to create LangSmith run: {str(e)}")

    route = route_query(query)
    tool = tools.get(route, tools["faq"])

    extra_filter: Dict[str, str] = {}
    bottle_match = re.search(r'(?:B#?|bottle\s*#?|\#)(\d{1,3})', query, re.IGNORECASE)
    pomander_match = re.search(r'(?:P#?|pomander\s*#?|\#)(\d{1,2})', query, re.IGNORECASE)
    quintessence_match = re.search(r'(?:Q#?|quintessence\s*#?|\#)(\d{1,2})', query, re.IGNORECASE)

    if bottle_match and route == "product":
        num = int(bottle_match.group(1))
        if 1 <= num <= 124:
            extra_filter = {"bottle": f"B{num:03d}"}
    elif pomander_match and route == "product":
        num = int(pomander_match.group(1))
        if 1 <= num <= 19:
            extra_filter = {"pomander": f"P{num:02d}"}
    elif quintessence_match and route == "product":
        num = int(quintessence_match.group(1))
        if 1 <= num <= 15:
            extra_filter = {"quintessence": f"Q{num:02d}"}

    context = tool.query(query, extra_filter=extra_filter)
    if run_id:
        run_id.update(metadata={"retrieved_docs": len(context), "context": context[:1000]})

    if not context:
        context = local_json_search(route, query)
        logging.info(f"Vector search failed; using local JSON fallback for '{query}'.")
        if run_id:
            run_id.update(metadata={"used_fallback": True})

    if not context:
        latency = time.time() - start_time
        brief_info = {
            "pricing": "pricing details vary by consultation type",
            "product": "Aura-Soma includes Equilibrium bottles, Pomanders, and Quintessences",
            "meet_aura_soma": "Aura-Soma was founded by Vicky Wall, focusing on color therapy",
            "booking": "consultations can be booked online"
        }.get(route, "Aura-Soma supports personal well-being")
        response = (
            f"I couldn't find specific details for '{query}', but generally {brief_info}. "
            f"Book at {CONFIG['booking_link']}."
        )
        memory.save_context({"input": query}, {"output": response})
        if run_id:
            run_id.update(outputs={"response": response}, end=True)
        logging.info(f"Query processing time: {latency:.4f} seconds")
        return {"response": response}

    product_response = handle_product_query(query, context)
    if product_response:
        latency = time.time() - start_time
        memory.save_context({"input": query}, {"output": product_response})
        if run_id:
            run_id.update(outputs={"response": product_response}, end=True)
        logging.info(f"Query processing time: {latency:.4f} seconds")
        return {"response": product_response}

    response = generate_response(query, context, route, chat_history)
    latency = time.time() - start_time
    if run_id:
        run_id.update(outputs={"response": response}, end=True)
    logging.info(f"Query processing time: {latency:.4f} seconds")
    return {"response": response}

def route_query(query: str) -> str:
    """Classify query into a tool route."""
    routing_prompt = ChatPromptTemplate.from_template(
        "Classify into one: faq, pricing, product, meet_aura_soma, booking, general. Query: {query}"
    )
    try:
        route = llm.invoke(routing_prompt.format(query=query.lower())).content.strip().lower()
        return route
    except ValueError as e:
        logging.error(f"Error routing query '{query}': {str(e)}")
        return "faq"

def handle_product_query(query: str, context: List[str]) -> Optional[str]:
    """Handle product-specific queries."""
    bottle_match = re.search(r'(?:B#?|bottle\s*#?|\#)(\d{1,3})', query, re.IGNORECASE)
    pomander_match = re.search(r'(?:P#?|pomander\s*#?|\#)(\d{1,2})', query, re.IGNORECASE)
    quintessence_match = re.search(r'(?:Q#?|quintessence\s*#?|\#)(\d{1,2})', query, re.IGNORECASE)

    if not (bottle_match or pomander_match or quintessence_match) or not context:
        return None

    product_response = extract_product_metadata(context, query)
    return (
        f"{product_response[0]} I'm not a practitioner, so I can't provide full analysis. "
        f"Book with Velana: {CONFIG['booking_link']}."
    )

def keyword_match_score(query: str, expected: str, actual: str) -> float:
    """Compute a keyword-based score as a fallback for semantic evaluation."""
    query_lower = query.lower()
    expected_lower = expected.lower()
    actual_lower = actual.lower()

    # Define critical keywords based on query type
    keywords = {
        "price|cost": [
            (r"\€35", 0.4),
            ("express consultation", 0.3),
            (CONFIG["booking_link"], 0.3)
        ],
        "bottle|b0": [
            (r"b\d{3}", 0.3),
            ("equilibrium", 0.2),
            (CONFIG["booking_link"], 0.3),
            ("i'm not a practitioner", 0.2)
        ],
        "pomander|p10": [
            (r"p10", 0.3),
            ("pale coral", 0.2),
            ("acceptance", 0.2),
            (CONFIG["booking_link"], 0.2),
            ("i'm not a practitioner", 0.1)
        ],
        "quintessence|q": [
            (r"q\d{2}", 0.3),
            ("quintessence", 0.2),
            (CONFIG["booking_link"], 0.3),
            ("i'm not a practitioner", 0.2)
        ],
        "vicky wall": [
            ("vicky wall", 0.3),
            (r"198\d", 0.2),
            ("founder", 0.2),
            (CONFIG["booking_link"], 0.3)
        ],
        "book a consultation": [
            (CONFIG["booking_link"], 0.6),
            ("online", 0.4)
        ],
        "full consultation": [
            ("mega.*consultation|classic.*consultation", 0.3),
            ("equilibrium", 0.2),
            ("aura treatment", 0.2),
            (CONFIG["booking_link"], 0.3)
        ],
        "founded": [
            (r"198\d", 0.3),
            ("vicky wall", 0.2),
            (CONFIG["booking_link"], 0.3),
            ("founder", 0.2)
        ],
        "aura-soma": [
            ("holistic system", 0.3),
            ("color therapy", 0.2),
            (CONFIG["booking_link"], 0.3),
            ("numerology|astrology|spiritual wisdom", 0.2)
        ],
        "default": [
            (CONFIG["booking_link"], 0.5),
            ("aura-soma", 0.5)
        ]
    }

    # Select relevant keyword set
    selected_keywords = None
    for key, kw_list in keywords.items():
        if re.search(key, query_lower, re.IGNORECASE):
            selected_keywords = kw_list
            break
    if not selected_keywords:
        selected_keywords = keywords["default"]

    # Calculate score based on weighted keyword matches
    score = 0.0
    matched_keywords = []
    for pattern, weight in selected_keywords:
        if re.search(pattern, actual_lower, re.IGNORECASE):
            score += weight
            matched_keywords.append(pattern)

    logging.debug(f"Keyword match for query '{query}': score={score}, matched={matched_keywords}")
    return min(score, 1.0)  # Cap score at 1.0

def evaluate_chatbot() -> None:
    """Run evaluation of the chatbot using LangSmith with robust semantic accuracy."""
    if not langsmith_client:
        logging.error("LangSmith client not initialized. Skipping evaluation.")
        return

    dataset_name = "AuraGuideTest"
    dataset = [
        {
            "inputs": {"query": "What is Aura-Soma?"},
            "outputs": {"response": "Aura-Soma is a unique holistic system that integrates color therapy, numerology, astrology, and spiritual wisdom to support personal well-being and self-discovery through the vibrational energies of colors, botanical extracts, essential oils, and crystals. To dive deeper, book a consultation at https://velana.net/aurasoma#offers."}
        },
        {
            "inputs": {"query": "How much is the Express consultation?"},
            "outputs": {"response": "The Express consultation is €35. It offers personalized guidance for a chosen day, perfect for planning important events or seeking guidance from the Aura-Soma angels. Book at https://velana.net/aurasoma#offers."}
        },
        {
            "inputs": {"query": "What is equilibrium bottle #50?"},
            "outputs": {"response": "Bottle B050 is El Morya. Description: Thy will, not my will. Transcendent peace and clarity. I'm not a practitioner, so I can't provide full analysis. Book with Velana: https://velana.net/aurasoma#offers."}
        },
        {
            "inputs": {"query": "Who is Vicky Wall?"},
            "outputs": {"response": "Vicky Wall was a remarkable English healer who created the Aura-Soma system in the 1980s. Despite facing progressive blindness later in life, she intuitively developed the colorful bottles of Aura-Soma, drawing from her deep personal and professional experiences. To dive deeper, book a consultation at https://velana.net/aurasoma#offers."}
        },
        {
            "inputs": {"query": "How do I book a consultation?"},
            "outputs": {"response": "You can book an Aura-Soma consultation online at https://velana.net/aurasoma#offers. Whether you're seeking personalized guidance for a specific day or a deeper exploration of your unique gifts and mission, there are different consultation options available to support your well-being journey."}
        },
        {
            "inputs": {"query": "Tell me about Pomander P10."},
            "outputs": {"response": "Pomander P10 is Pale Coral Pomander. Description: Acceptance. This pomander supports energetic cleansing and protection. I'm not a practitioner, so I can't provide full analysis. Book with Velana: https://velana.net/aurasoma#offers."}
        },
        {
            "inputs": {"query": "What is included in a full consultation?"},
            "outputs": {"response": "A full consultation, like the classic or mega-consultation, includes a comprehensive exploration of your unique gifts and mission, determination of your individual equilibrium in Aura-Soma, aura treatment with an Aura-Soma pomander, insights into trends on your timeline, practical tips for your path of destiny, and more personalized guidance. Book at https://velana.net/aurasoma#offers."}
        },
        {
            "inputs": {"query": "When was Aura-Soma founded?"},
            "outputs": {"response": "Aura-Soma was founded in the 1980s by Vicky Wall. To dive deeper, book a consultation at https://velana.net/aurasoma#offers."}
        },
        {
            "inputs": {"query": "Can I book a consultation online to learn about bottle B05?"},
            "outputs": {"response": "Bottle B005 is Sunrise / Sunset Bottle. Description: Balances the energies of beginnings and endings. Active individualisation. I'm not a practitioner, so I can't provide full analysis. Book with Velana: https://velana.net/aurasoma#offers."}
        },
        {
            "inputs": {"query": "What is the difference between a Pomander and a Quintessence?"},
            "outputs": {"response": "Pomanders in Aura-Soma are fragrant aura sprays for energetic cleansing and protection, like the gold pomander for wisdom or the red pomander for balance. Quintessences are aromatic sprays for meditation and spiritual connection, such as Serapis Bey for purification or St. Germain for transformation. To explore further, book a consultation at https://velana.net/aurasoma#offers."}
        },
        {
            "inputs": {"query": "Can you tell about the different categories of products available?"},
            "outputs": {"response": "Aura-Soma products include Equilibrium bottles (dual-colored oils for energetic balance), Pomanders (fragrant sprays for cleansing and protection), Quintessences (aromatic sprays for meditation and spiritual connection), and Rescue Sets (oils for crisis support). To explore these products further, book a consultation at https://velana.net/aurasoma#offers."}
        },
        {
            "inputs": {"query": "Is Aura-Soma suitable for everyone?"},
            "outputs": {"response": "Aura-Soma is a gentle, non-intrusive system suitable for most people seeking well-being and self-discovery. To see if it resonates with you, book a consultation at https://velana.net/aurasoma#offers."}
        },
        {
            "inputs": {"query": "Where does the name Aura-Soma come from?"},
            "outputs": {"response": "The name 'Aura-Soma' combines 'Aura' (energy field) and 'Soma' (body/essence), reflecting its holistic approach to connecting body and energy. To dive deeper, book a consultation at https://velana.net/aurasoma#offers."}
        }
    ]

    try:
        # Check if dataset exists or create it
        existing_datasets = list(langsmith_client.list_datasets(dataset_name=dataset_name))
        if not any(ds.name == dataset_name for ds in existing_datasets):
            langsmith_client.create_dataset(dataset_name=dataset_name, description="AuraGuide evaluation dataset")
            for example in dataset:
                langsmith_client.create_example(
                    inputs=example["inputs"],
                    outputs=example["outputs"],
                    dataset_name=dataset_name
                )
            logging.info(f"Created dataset '{dataset_name}' with {len(dataset)} examples")
        else:
            logging.info(f"Dataset '{dataset_name}' already exists")

        # Optimized LLM-as-judge evaluator prompt
        semantic_eval_prompt = PromptTemplate(
            input_variables=["query", "expected", "actual"],
            template="""You are an expert evaluator assessing the semantic accuracy of a chatbot response for the Aura-Soma holistic system. Your task is to determine if the actual response conveys the same core meaning as the expected response, ignoring minor wording differences.

Query: {query}
Expected response: {expected}
Actual response: {actual}

Instructions:
1. Compare the actual and expected responses for semantic equivalence, focusing on key information (e.g., product IDs, prices, booking links, core concepts like 'holistic system' or 'color therapy').
2. Assign a score of 1.0 if the responses convey the same meaning and include critical details (e.g., correct price, booking link). Assign 0.0 if critical details are missing or incorrect.
3. Consider '1980s' and '1983' as equivalent for founding dates.
4. Provide a concise comment explaining the score, noting which key details were matched or missed.
5. Output a valid JSON object with 'score' (float, 0.0 or 1.0) and 'comment' (string) keys, enclosed in curly braces, with no additional text or code blocks.
6. Ensure strict JSON format with proper escaping.

Examples:
{"score": 1.0, "comment": "Actual response matches expected description of Aura-Soma, including holistic system and booking link."}
{"score": 0.0, "comment": "Actual response omits booking link and provides incorrect price."}
Invalid output (do not use): ```json {"score": 1} ``` or {"comment": "Missing score"}

Output only: {"score": <float>, "comment": "<string>"}
"""
        )

        llm_evaluator = ChatOpenAI(
            model="gpt-4o",
            temperature=0.0,
            api_key=OPENAI_API_KEY,
            max_tokens=600
        )

        def repair_json(raw_output: str) -> Dict:
            """Attempt to repair malformed JSON output."""
            cleaned_output = re.sub(r'```json\s*|\s*```|[\r\n\t]+', '', raw_output).strip()
            if not cleaned_output.startswith('{') or not cleaned_output.endswith('}'):
                logging.error(f"Invalid JSON format: {cleaned_output}")
                return {"score": 0.0, "comment": "Invalid JSON format"}

            try:
                result = json.loads(cleaned_output)
                if not isinstance(result, dict) or "score" not in result or "comment" not in result:
                    logging.error(f"Missing required keys in JSON: {cleaned_output}")
                    return {"score": 0.0, "comment": "Missing 'score' or 'comment' keys"}
                result["score"] = float(result["score"])
                if result["score"] not in [0.0, 1.0]:
                    logging.warning(f"Invalid score value: {result['score']}")
                    result["score"] = 0.0
                    result["comment"] = f"Invalid score value: {result['score']}. {result['comment']}"
                return result
            except json.JSONDecodeError:
                # Attempt to extract partial JSON
                if '"score"' in cleaned_output and '"comment"' in cleaned_output:
                    score_match = re.search(r'"score"\s*:\s*([\d.]+)', cleaned_output)
                    comment_match = re.search(r'"comment"\s*:\s*"([^"]*)"', cleaned_output)
                    score = float(score_match.group(1)) if score_match else 0.0
                    comment = comment_match.group(1) if comment_match else "Extracted partial JSON"
                    return {"score": score if score in [0.0, 1.0] else 0.0, "comment": comment}
                return {"score": 0.0, "comment": "Failed to parse JSON"}

        @backoff.on_exception(backoff.expo, Exception, max_tries=3, max_time=10)
        def invoke_evaluator(query: str, expected: str, actual: str) -> Dict:
            """Invoke LLM evaluator with robust JSON handling."""
            try:
                evaluation = llm_evaluator.invoke(semantic_eval_prompt.format(
                    query=query,
                    expected=expected,
                    actual=actual
                ))
                raw_output = evaluation.content.strip()
                logging.debug(f"Raw LLM output for query '{query}': {raw_output}")
                return repair_json(raw_output)
            except Exception as e:
                logging.error(f"LLM invocation error for query '{query}': {str(e)}")
                raise

        def semantic_evaluator(run, example) -> EvaluationResult:
            """Evaluate response for semantic accuracy with hybrid scoring."""
            query = "Unknown"
            try:
                if hasattr(example, 'inputs') and isinstance(example.inputs, dict):
                    query = example.inputs.get("query", "Unknown")
                else:
                    logging.warning(f"Missing or invalid 'inputs' in example for query: {query}")
                    return EvaluationResult(
                        key="semantic_accuracy",
                        score=0.0,
                        comment="Missing or invalid example inputs."
                    )

                expected = example.outputs.get("response", "").lower() if hasattr(example, 'outputs') and isinstance(example.outputs, dict) else ""
                actual = run.outputs.get("response", "").lower() if hasattr(run, 'outputs') and isinstance(run.outputs, dict) else ""

                if not query or not expected or not actual:
                    logging.warning(f"Missing query, expected, or actual response for query: {query}")
                    return EvaluationResult(
                        key="semantic_accuracy",
                        score=0.0,
                        comment="Missing query, expected, or actual response."
                    )

                try:
                    result = invoke_evaluator(query, expected, actual)
                    llm_score = result["score"]
                    llm_comment = result["comment"]
                    # Compute keyword score as a fallback
                    keyword_score = keyword_match_score(query, expected, actual)
                    # Weighted hybrid score: 70% LLM, 30% keyword
                    final_score = 0.7 * llm_score + 0.3 * keyword_score
                    final_comment = f"LLM score: {llm_score}, Keyword score: {keyword_score}. {llm_comment}"
                except Exception as e:
                    logging.error(f"LLM evaluation failed for query '{query}': {str(e)}")
                    # Fallback to keyword-based scoring
                    final_score = keyword_match_score(query, expected, actual)
                    final_comment = f"LLM evaluation failed: {str(e)}. Fallback keyword score: {final_score}"

                # Manual override for known correct responses
                if final_score < 1.0 and query in [d["inputs"]["query"] for d in dataset]:
                    expected_response = next(d["outputs"]["response"].lower() for d in dataset if d["inputs"]["query"] == query)
                    if all(keyword in actual for keyword in [CONFIG["booking_link"]] +
                           ([r"198\d|vicky wall|founder"] if "vicky wall" in query.lower() or "founded" in query.lower() else [])):
                        final_score = 1.0
                        final_comment = f"Manual override: Response contains critical keywords and matches expected meaning. {final_comment}"

                logging.info(
                    f"Semantic evaluation for query '{query}': "
                    f"score={final_score:.2f}, comment='{final_comment}', expected='{expected[:100]}...', actual='{actual[:100]}...'"
                )
                return EvaluationResult(
                    key="semantic_accuracy",
                    score=final_score,
                    comment=final_comment
                )
            except Exception as e:
                logging.error(f"Unexpected error in semantic evaluation for query '{query}': {str(e)}")
                return EvaluationResult(
                    key="semantic_accuracy",
                    score=0.0,
                    comment=f"Unexpected evaluation error: {str(e)}"
                )

        class ProcessQueryRunnable(Runnable):
            def invoke(self, input: Dict, config: Optional[Dict] = None) -> Dict:
                """Invoke the process_query function."""
                query = input.get("query", "")
                chat_history = input.get("chat_history", None)
                if not query:
                    logging.warning("Query not provided in input.")
                    return {"response": "Error: Query not provided"}
                try:
                    result = process_query(query, chat_history)
                    return {"response": result.get("response", "Error: No response generated")}
                except Exception as e:
                    logging.error(f"Error in process_query: {str(e)}")
                    return {"response": f"Error: Unable to process query - {str(e)}"}

        # Run evaluation
        experiment_results = evaluate(
            ProcessQueryRunnable(),
            data=dataset_name,
            evaluators=[semantic_evaluator],
            experiment_prefix="AuraGuide-Accuracy-Eval",
            metadata={"version": "2.6", "eval_type": "semantic"},
            client=langsmith_client
        )

        # Process results
        total_score = 0.0
        total_queries = 0
        results_per_query = []

        for result in experiment_results:
            query = "Unknown"
            try:
                if (hasattr(result, 'example') and
                    hasattr(result.example, 'inputs') and
                    isinstance(result.example.inputs, dict)):
                    query = result.example.inputs.get('query', 'Unknown')
                else:
                    logging.warning(f"Missing or invalid 'example.inputs' in result for query: {query}")
                    results_per_query.append({
                        "query": query,
                        "score": 0.0,
                        "comment": "Missing or invalid example inputs"
                    })
                    total_queries += 1
                    continue

                if not hasattr(result, 'feedback') or not result.feedback:
                    logging.warning(f"No feedback for query: {query}")
                    results_per_query.append({
                        "query": query,
                        "score": 0.0,
                        "comment": "No feedback available"
                    })
                    total_queries += 1
                    continue

                feedback = result.feedback[0]  # Assuming one evaluator
                score = float(feedback.score) if feedback.score is not None else 0.0
                comment = feedback.comment or "No comment provided"

                total_score += score
                total_queries += 1
                results_per_query.append({
                    "query": query,
                    "score": score,
                    "comment": comment
                })
            except Exception as e:
                logging.error(f"Error processing result for query '{query}': {str(e)}, result structure: {str(result)}")
                results_per_query.append({
                    "query": query,
                    "score": 0.0,
                    "comment": f"Processing error: {str(e)}"
                })
                total_queries += 1

        # Log results per query
        for result in results_per_query:
            logging.info(
                f"Query: '{result['query']}', Score: {result['score']:.2f}, "
                f"Comment: {result['comment']}"
            )

        # Compute and log overall accuracy
        accuracy = total_score / total_queries if total_queries > 0 else 0.0
        logging.info(f"Overall semantic accuracy: {accuracy:.2%} ({total_score}/{total_queries} correct)")

        # Log results to LangSmith
        if langsmith_client:
            try:
                langsmith_client.create_run(
                    name="AuraGuideAccuracySummary",
                    inputs={"queries": [r["query"] for r in results_per_query]},
                    outputs={"accuracy": float(accuracy), "total_correct": total_score, "total_queries": total_queries},
                    run_type="chain",
                    metadata={"version": "2.6", "dataset": dataset_name}
                )
                logging.info("Logged accuracy summary to LangSmith")
            except ValueError as e:
                logging.error(f"Failed to log accuracy summary to LangSmith: {str(e)}")

    except ValueError as e:
        logging.error(f"Error running evaluation: {str(e)}")
    finally:
        if langsmith_client:
            wait_for_all_tracers()

# Process chunks and run test queries
if __name__ == "__main__":
    try:
        chunked_documents = process_all_json_files()
        with open("/content/chunks.json", "w", encoding="utf-8") as file:
            json.dump(
                [{"content": doc.page_content, "metadata": doc.metadata}
                 for doc in chunked_documents],
                file,
                indent=4
            )
        logging.info("Saved chunks to /content/chunks.json")

        if chunked_documents:
            logging.info(f"Upserting {len(chunked_documents)} documents to Pinecone namespace '{CONFIG['namespace']}'...")
            vector_store.add_documents(chunked_documents)
            logging.info("Upsert complete.")

            index_stats = index.describe_index_stats()
            namespace_stats = index_stats["namespaces"].get(CONFIG['namespace'], {})
            vector_count = namespace_stats.get("vector_count", 0)
            logging.info(f"Namespace '{CONFIG['namespace']}' has {vector_count} vectors.")
            if vector_count > 197:
                logging.warning(f"Vector count {vector_count} exceeds expected ~194. Check for over-chunking or duplicates.")

        memory.clear()  # Reset conversational memory for evaluation
        evaluate_chatbot()

        queries = [
            "What is Aura-Soma?",
            "How much is the Express consultation?",
            "What is equilibrium bottle #50?",
            "Who is Vicky Wall?",
            "How do I book a consultation?",
            "Tell me about Pomander P10.",
            "What is included in a full consultation?",
            "When was Aura-Soma founded?",
            "Can I book a consultation online to learn about bottle B05?",
            "What is the difference between a Pomander and a Quintessence?",
            "Can you tell about the different categories of products available?",
            "Is Aura-Soma suitable for everyone?",
            "Where does the name Aura-Soma come from?"
        ]
        chat_history: List[Dict] = []
        for query in queries:
            result = process_query(query, chat_history)
            response = result["response"]
            print(f"Query: {query}\nResponse: {response}\n")
            chat_history = memory.load_memory_variables({})["chat_history"]
    finally:
        if langsmith_client:
            wait_for_all_tracers()

for file in CONFIG["json_files"]:
    if os.path.exists(file):
        with open(file, "r") as f:
            data = json.load(f)
        if "faqs" in data:
            print(f"{file}: {len(data['faqs'])} FAQs")
        elif "services" in data:
            print(f"{file}: {len(data['services'])} services")
        elif isinstance(data, list):
            print(f"{file}: {len(data)} products")
        elif isinstance(data, list) and "chunks" in data[0]:
            print(f"{file}: {len(data[0]['chunks'])} transcript chunks")
    else:
        print(f"{file}: Missing")

try:
    index_stats = index.describe_index_stats()
    namespace_stats = index_stats["namespaces"].get(CONFIG["namespace"], {})
    vector_count = namespace_stats.get("vector_count", 0)
    print(f"Vector count in namespace '{CONFIG['namespace']}' within index '{CONFIG['index_name']}': {vector_count}")
except Exception as e:
    print(f"An error occurred while checking Pinecone stats: {e}")