import streamlit as st
import os
from datetime import datetime
from typing import List, Dict, Optional, Callable  # Added Callable
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
import re
import json
import logging
import time
from functools import partial

# --- Configure logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- API Key Security: Use Streamlit secrets ---
try:
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    INDEX_NAME = st.secrets["INDEX_NAME"]  # Ensure this is "aura-soma"
except KeyError as e:
    st.error(f"Missing secret key: {e}. Please set it in Streamlit secrets.")
    st.stop()

# --- Validate keys and index ---
if not PINECONE_API_KEY or not INDEX_NAME:
    st.error("PINECONE_API_KEY or INDEX_NAME is not set properly.")
    st.stop()

# --- CONFIG from Colab, adapted for app.py ---
CONFIG = {
    "booking_link": "https://velana.net/aurasoma#offers",
    "search_k": 5,
    "similarity_threshold": 0.1,
    "index_name": "aura-soma",
    "namespace": "aura-soma-velana",
    "vector_store": "aura-soma-vs1",
    "embedding_dimensions": 1024,
    "embedding_model": "text-embedding-3-large",
    # Use chunks.json for all routes as fallback
    "json_sources": {
        "faq": "chunks.json",
        "pricing": "chunks.json",
        "product": "chunks.json",
        "meet_aura_soma": "chunks.json",
        "booking": "chunks.json"
    }
}

# --- Cache Pinecone initialization ---
@st.cache_resource
def init_pinecone():
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        if INDEX_NAME not in [idx["name"] for idx in pc.list_indexes().get("indexes", [])]:
            st.error(f"Index {INDEX_NAME} does not exist in us-east-1.")
            st.stop()
        return pc
    except Exception as e:
        st.error(f"Pinecone initialization failed: {str(e)}")
        st.stop()

# --- Cache embeddings ---
@st.cache_resource
def get_embeddings():
    try:
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        return OpenAIEmbeddings(model=CONFIG["embedding_model"], dimensions=CONFIG["embedding_dimensions"])
    except Exception as e:
        st.error(f"OpenAI Embeddings failed: {str(e)}")
        st.stop()

# --- Initialize Pinecone and Vector Store ---
@st.cache_resource
def initialize_pinecone():
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        existing_indexes = pc.list_indexes().names()
        if INDEX_NAME not in existing_indexes:
            logger.info(f"Creating Pinecone index: {INDEX_NAME}")
            pc.create_index(
                name=INDEX_NAME,
                dimension=CONFIG["embedding_dimensions"],
                metric="cosine",
                spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
            )
            while not pc.describe_index(INDEX_NAME).status["ready"]:
                time.sleep(1)
            logger.info(f"Index '{INDEX_NAME}' created and ready.")
        else:
            logger.info(f"Index '{INDEX_NAME}' already exists. Connecting...")

        index = pc.Index(INDEX_NAME)
        index_stats = index.describe_index_stats()
        if index_stats["dimension"] != CONFIG["embedding_dimensions"]:
            raise ValueError(
                f"Index dimension {index_stats['dimension']} does not match "
                f"embedding dimension {CONFIG['embedding_dimensions']}"
            )

        # Optional: Clear namespace (comment out to avoid data loss - preserving 198 vectors)
        # for attempt in range(3):
        #     logger.info(f"Clearing namespace '{CONFIG['namespace']}' (attempt {attempt + 1})...")
        #     try:
        #         index.delete(delete_all=True, namespace=CONFIG["namespace"])
        #         time.sleep(2)
        #         index_stats = index.describe_index_stats()
        #         vector_count = index_stats["namespaces"].get(CONFIG["namespace"], {}).get("vector_count", 0)
        #         if vector_count == 0:
        #             logger.info(f"Namespace '{CONFIG['namespace']}' successfully cleared.")
        #             break
        #     except Exception as e:
        #         logger.warning(f"Namespace deletion attempt {attempt + 1} failed: {str(e)}")
        # else:
        #     logger.warning(f"Namespace '{CONFIG['namespace']}' not found or failed to clear. Proceeding.")

        embeddings = get_embeddings()
        vector_store = PineconeVectorStore(
            index=index,
            embedding=embeddings,
            namespace=CONFIG["namespace"],
            text_key=CONFIG["vector_store"]
        )
        return vector_store, index
    except Exception as e:
        st.error(f"Vector store initialization error: {str(e)}")
        st.stop()

vector_store, index = initialize_pinecone()

# --- Vector Query Tool ---
class VectorQueryTool:
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
            logger.info(
                f"{self.__class__.__name__} retrieved {len(filtered)} docs for query "
                f"'{query}' with filter {filter_dict}. Scores: {[score for _, score in filtered]}"
            )
            documents = [doc for doc, _ in filtered]
            if not documents and fallback_to_general and filter_dict:
                logger.info(f"No results with filter for '{query}'. Trying general search.")
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
                logger.info(
                    f"General search retrieved {len(filtered)} docs for '{query}'. "
                    f"Scores: {[score for _, score in filtered]}"
                )
                documents = [doc for doc, _ in filtered]
            if self.post_process:
                documents = self.post_process(documents, query)
            return documents
        except Exception as e:
            logger.error(f"Vector store query error for '{query}': {str(e)}")
            return []

# --- Product Metadata Extraction ---
def build_product_pattern(id_prefix: str, num: int, category: str) -> re.Pattern:
    """Build regex pattern for product extraction."""
    num_str = f"{num:03d}" if id_prefix == "B" else f"{num:02d}"
    return re.compile(
        rf'"{id_prefix}{num_str}".*?"name":"(.*?)".*?"category":"{category}".*?"description":"(.*?)"',
        re.IGNORECASE | re.DOTALL
    )

def extract_product_metadata(documents: List[str], query: str) -> List[str]:
    bottle_match = re.search(r'(?:B#?|bottle\s*#?|\#)(\d{1,3})', query, re.IGNORECASE)
    pomander_match = re.search(r'(?:P#?|pomander\s*#?|\#)(\d{1,2})', query, re.IGNORECASE)
    quintessence_match = re.search(r'(?:Q#?|quintessence\s*#?|\#)(\d{1,2})', query, re.IGNORECASE)

    short_descriptions = []

    # Try vector store results first
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
            # Fallback message
            return [f"Bottle {bottle_id} exists in the Aura-Soma Equilibrium range. Please consult a practitioner for detailed insights."]
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
                        logger.info(f"P10 details: name={name}, desc={desc}")
                    return [f"Pomander {pomander_id} is {name}. Description: {desc}."]
            return [f"Pomander {pomander_id} exists in the Aura-Soma range. Please consult a practitioner for details."]
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
            return [f"Quintessence {quintessence_id} exists in the Aura-Soma range. Please consult a practitioner for details."]
    return documents

    # Log short descriptions if any
    if short_descriptions:
        logger.warning(f"Short descriptions found: {short_descriptions}. Consider updating Aura_Soma_Products.json.")

    # Fallback to direct JSON search using chunks.json
    file_path = CONFIG["json_sources"]["product"]
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            json_data = json.load(file)
        if not isinstance(json_data, list):
            logger.error(f"Expected list in {file_path}, got {type(json_data)}")
            return []
        if bottle_match:
            num = int(bottle_match.group(1))
            if 1 <= num <= 124:
                bottle_id = f"B{num:03d}"
                for product in json_data:
                    if product.get("id") == bottle_id:
                        desc = product.get("description", "")
                        if len(desc) < 20:
                            short_descriptions.append(f"{bottle_id}: {desc}")
                            desc += " This bottle supports energetic balance."
                        return [f"Bottle {bottle_id} is {product.get('name', 'Unknown')}. Description: {desc}."]
                return [f"Bottle {bottle_id} exists. Consult a practitioner for details at {CONFIG['booking_link']}."]
        elif pomander_match:
            num = int(pomander_match.group(1))
            if 1 <= num <= 19:
                pomander_id = f"P{num:02d}"
                for product in json_data:
                    if product.get("id") == pomander_id:
                        desc = product.get("description", "")
                        if len(desc) < 20:
                            short_descriptions.append(f"{pomander_id}: {desc}")
                            desc += " This pomander supports cleansing."
                        return [f"Pomander {pomander_id} is {product.get('name', 'Unknown')}. Description: {desc}."]
                return [f"Pomander {pomander_id} exists. Consult a practitioner at {CONFIG['booking_link']}."]
        elif quintessence_match:
            num = int(quintessence_match.group(1))
            if 1 <= num <= 15:
                quintessence_id = f"Q{num:02d}"
                for product in json_data:
                    if product.get("id") == quintessence_id:
                        desc = product.get("description", "")
                        if len(desc) < 20:
                            short_descriptions.append(f"{quintessence_id}: {desc}")
                            desc += " This quintessence supports meditation."
                        return [f"Quintessence {quintessence_id} is {product.get('name', 'Unknown')}. Description: {desc}."]
                return [f"Quintessence {quintessence_id} exists. Consult a practitioner at {CONFIG['booking_link']}."]
    except FileNotFoundError:
        logger.error(f"JSON file not found: {file_path}")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {file_path}: {str(e)}")
    return documents

def append_booking_redirect(documents: List[str], query: str) -> List[str]:
    """Append booking link to retrieved documents."""
    return documents + [
        f"To book a consultation or purchase a gift card, visit: {CONFIG['booking_link']}"
    ]

 
 def local_json_search(route: str, query: str) -> List[str]:
     try:
         with open(CONFIG["json_sources"][route], "r", encoding="utf-8") as file:
             json_data = json.load(file)
         matches = []
         query_lower = query.lower()
         if route == "meet_aura_soma":
             for chunk in json_data:
                 if "vicky wall" in chunk.get("content", "").lower():
                     matches.append(chunk.get("content", ""))
             if not matches:
                 matches.append("Vicky Wall (1918-1991) founded Aura-Soma in 1983. A healer and pharmacologist, she created it through intuitive vision. To dive deeper, book at {CONFIG['booking_link']}.")
         elif route == "product":
             bottle_match = re.search(r'(?:b#?|bottle\s*#?|\#)(\d{1,3})', query_lower, re.IGNORECASE)
             pomander_match = re.search(r'(?:p#?|pomander\s*#?|\#)(\d{1,2})', query_lower, re.IGNORECASE)
             quintessence_match = re.search(r'(?:q#?|quintessence\s*#?|\#)(\d{1,2})', query_lower, re.IGNORECASE)
             if bottle_match:
                 num = int(bottle_match.group(1))
                 bottle_id = f"b{num:03d}"
                 for chunk in json_data:
                     if bottle_id in chunk.get("content", "").lower():
                         matches.append(chunk.get("content", ""))
             elif pomander_match:
                 num = int(pomander_match.group(1))
                 pomander_id = f"p{num:02d}"
                 for chunk in json_data:
                     if pomander_id in chunk.get("content", "").lower():
                         matches.append(chunk.get("content", ""))
             elif quintessence_match:
                 num = int(quintessence_match.group(1))
                 quintessence_id = f"q{num:02d}"
                 for chunk in json_data:
                     if quintessence_id in chunk.get("content", "").lower():
                         matches.append(chunk.get("content", ""))
             if not matches:
                 matches.append("No product details. Includes Equilibrium bottles.")
         elif route == "booking":
             for chunk in json_data:
                 if "book" in chunk.get("content", "").lower():
                     matches.append(chunk.get("content", ""))
             if not matches:
                 matches.append(f"Book your consultation at {CONFIG['booking_link']}.")
         elif route == "pricing":
             for chunk in json_data:
                 if "price" in chunk.get("content", "").lower():
                     matches.append(chunk.get("content", ""))
             if not matches:
                 matches.append("Consultations: Express €35, Classic €100, Mega €130. Book at {CONFIG['booking_link']}.")
         elif route == "faq":
             for chunk in json_data:
                 if query_lower in chunk.get("content", "").lower():
                     matches.append(chunk.get("content", ""))
             if not matches:
                 matches.append("No general info. Aura-Soma is a holistic system.")
         if not matches:
             brief_info = {
                 "pricing": "varies by service",
                 "product": "includes bottles, pomanders, quintessences",
                 "meet_aura_soma": "founded by Vicky Wall",
                 "booking": "book online",
                 "faq": "general Aura-Soma info"
             }.get(route, "general wellbeing")
             matches.append(f"No details for '{query}'. Generally {brief_info}. Book at {CONFIG['booking_link']}.")
         return matches
     except FileNotFoundError:
         logger.error(f"chunks.json not found")
         return [f"File missing for '{query}'. Book at {CONFIG['booking_link']}."]
     except json.JSONDecodeError as e:
         logger.error(f"Invalid JSON: {str(e)}")
         return [f"Invalid data for '{query}'. Book at {CONFIG['booking_link']}."]

def initialize_pinecone():
    """Initialize Pinecone with safe namespace clearing and verification."""
    try:
        pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
        index_name = CONFIG["index_name"]
        namespace = CONFIG["namespace"]

        existing_indexes = pinecone_client.list_indexes().names()
        if index_name not in existing_indexes:
            logger.info(f"Creating Pinecone index: {index_name}")
            pinecone_client.create_index(
                name=index_name,
                dimension=CONFIG["embedding_dimensions"],
                metric="cosine",
                spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
            )
            while not pinecone_client.describe_index(index_name).status["ready"]:
                time.sleep(1)
            logger.info(f"Index '{index_name}' created and ready.")
        else:
            logger.info(f"Index '{index_name}' already exists. Connecting...")

        index = pinecone_client.Index(index_name)
        index_stats = index.describe_index_stats()
        if index_stats["dimension"] != CONFIG["embedding_dimensions"]:
            raise ValueError(
                f"Index dimension {index_stats['dimension']} does not match "
                f"embedding dimension {CONFIG['embedding_dimensions']}"
            )

        # Optional: Clear namespace (comment out to avoid data loss - preserving 198 vectors)
        # for attempt in range(3):
        #     logger.info(f"Clearing namespace '{namespace}' (attempt {attempt + 1})...")
        #     try:
        #         index.delete(delete_all=True, namespace=namespace)
        #         time.sleep(2)
        #         index_stats = index.describe_index_stats()
        #         vector_count = index_stats["namespaces"].get(namespace, {}).get("vector_count", 0)
        #         if vector_count == 0:
        #             logger.info(f"Namespace '{namespace}' successfully cleared.")
        #             break
        #     except Exception as e:
        #         logger.warning(f"Namespace deletion attempt {attempt + 1} failed: {str(e)}")
        # else:
        #     logger.warning(f"Namespace '{namespace}' not found or failed to clear. Proceeding with ingestion.")

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
        logger.error(f"Vector store initialization error: {str(e)}")
        raise
    except KeyError as e:
        logger.error(f"Missing environment variable: {str(e)}")
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
        post_process=partial(extract_product_metadata)
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
        post_process=partial(append_booking_redirect)
    )
}

# --- Initialize LLM and Memory ---
memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history", max_len=10)

llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, api_key=OPENAI_API_KEY)

# --- System Prompt ---
system_prompt = """You are AuraGuide, a mindful chatbot for Velana.net, specializing in Aura-Soma. Respond concisely, warmly, with light and harmony. Be positive and non-judgmental.

system_prompt = """You are AuraGuide, a helpful, mindful chatbot for Velana.net, specializing in Aura-Soma—a holistic color-care system for wellbeing. Respond concisely, warmly, using language inspired by light and harmony. Be positive, non-judgemental, and emphasize personal resonance.

Key Rules:
- Use provided {context} if relevant; otherwise, use tools or general knowledge.
- For general Aura-Soma questions, including shipping and returns, use FAQTool or MeetAuraSoma tools.
- For bottle/pomander/quintessence queries, use ProductTool to look at Aura_Soma_Products.json for id, name, category, and description, to extract name/title (e.g., "B001 - Blue/Deep Magenta - Physical Rescue"). Then say: "I'm not a practitioner, so I can't provide full analysis. Book with Velana: {booking_link}." 
- For pricing/services/consultation questions, use PricingTool to look at aura_soma_pricelist.json for name, price, description, or FAQTool. Bottles can be bought ideally after a personal consultation to ensure product fitness. Pomanders and Quintessence cost 27€ per bottle while Equilibrium costs 47€ per bottle. End with: "To dive deeper, book a consultation at {booking_link}."
- If no relevant info, say: "I couldn't find specific details, but generally [brief info]."
- Redirect to {booking_link} only for personalized advice in booking or services queries.
- Maintain conversational memory.

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
        "chat_history": chat_history or memory.load_memory_variables({})["chat_history"],
        "booking_link": CONFIG["booking_link"]
    }).content

    memory.save_context({"input": query}, {"output": response})
    return response

def process_query(query: str, chat_history: Optional[List[Dict]] = None) -> Dict:
    """Process a user query through the RAG pipeline."""
    start_time = time.time()
    if not query.strip():
        latency = time.time() - start_time
        response = f"Please provide a query. To dive deeper, book at {CONFIG['booking_link']}."
        logger.info(f"Query processing time: {latency:.4f} seconds")
        return {"response": response}

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

    if not context:
        context = local_json_search(route, query)
        logger.info(f"Vector search failed; using local JSON fallback for '{query}'.")

    if not context:
        latency = time.time() - start_time
        brief_info = {
            "pricing": "pricing details vary by consultation type",
            "product": "Aura-Soma includes Equilibrium bottles, Pomanders, and Quintessences",
            "meet_aura_soma": "Aura-Soma was founded by Vicky Wall",
            "booking": "consultations can be booked online"
        }.get(route, "Aura-Soma supports personal well-being")
        response = f"No specific details for '{query}'. Generally {brief_info}. Book at {CONFIG['booking_link']}."
        memory.save_context({"input": query}, {"output": response})
        logger.info(f"Query processing time: {latency:.4f} seconds")
        return {"response": response}

    product_response = handle_product_query(query, context)
    if product_response:
        latency = time.time() - start_time
        memory.save_context({"input": query}, {"output": product_response})
        logger.info(f"Query processing time: {latency:.4f} seconds")
        return {"response": product_response}

    response = generate_response(query, context, route, chat_history or [])
    latency = time.time() - start_time
    logger.info(f"Query processing time: {latency:.4f} seconds")
    return {"response": response}

def route_query(query: str) -> str:
    """Classify query into a tool route."""
    routing_prompt = ChatPromptTemplate.from_template(
        """Classify into one: faq, pricing, product, meet_aura_soma, booking, general.
        - 'faq' for general questions (e.g., 'What is Aura-Soma?', 'shipping').
        - 'pricing' for price or service questions (e.g., 'How much?', 'consultation cost').
        - 'product' for product queries (e.g., 'What is B15?', 'P10 Pomander').
        - 'meet_aura_soma' for history questions (e.g., 'Who is Vicky Wall?').
        - 'booking' for booking-related questions (e.g., 'How do I book?').
        - 'general' for unclassifiable queries.
        Query: {query}"""
    )
    try:
        route = llm.invoke(routing_prompt.format(query=query.lower())).content.strip().lower()
        return route if route in ["faq", "pricing", "product", "meet_aura_soma", "booking", "general"] else "faq"
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

# --- Cache chunks.json load ---
@st.cache_data
def load_chunks():
    try:
        with open("chunks.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("chunks.json not found in repo root. Please ensure it’s committed to GitHub.")
        st.stop()
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON in chunks.json: {str(e)}. Please check the file format.")
        st.stop()

# --- Streamlit UI ---
st.set_page_config(page_title="Aura Guide Bot", layout="wide")

# CSS for avatars and polished chat bubbles
st.markdown(
    """
    <style>
    .chat-container {
        max-height: 70vh;
        overflow-y: auto;
        padding: 10px;
        display: flex;
        flex-direction: column;
    }
    .chat-row {
        display: flex;
        margin: 5px 0;
        align-items: flex-start;
    }
    .user-msg, .bot-msg {
        padding: 10px;
        border-radius: 15px;
        max-width: 75%;
        word-wrap: break-word;
        color: #000000; /* Explicit black text for readability */
    }
    .user-msg {
        background-color: #DCF8C6;
        margin-left: 10px;
        order: 2;
    }
    .bot-msg {
        background-color: #EAEAEA;
        margin-right: 10px;
        order: 1;
    }
    .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
    }
    .timestamp {
        font-size: 0.7em;
        color: #888888;
        margin-top: 2px;
    }
    .user-row {
        flex-direction: row-reverse;
    }
    </style>
    """, unsafe_allow_html=True
)

st.title("Aura Guide Bot")

# --- Session state ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- Intro message ---
st.write("👋 I am graced by your presence! Ask me about Aura-Soma.")

# --- Sidebar settings ---
with st.sidebar:
    st.header("Settings")
    model_choice = st.selectbox("Choose model:", ["gpt-3.5-turbo-0125", "gpt-4"])
    temp = st.slider("Temperature:", 0.0, 1.0, 0.0)
    llm.model_name = model_choice
    llm.temperature = temp

# --- Clear chat button ---
if st.button("🗑️ Clear Chat"):
    st.session_state.history = []
    memory.clear()
    st.rerun()

# --- Load chunks for fallback ---
chunks = load_chunks()

# --- User input with validation ---
user_question = st.text_input("Type your question and press Enter:")
if user_question:
    user_question = user_question.strip()[:500].lower()
    if not user_question:
        st.warning("Please enter a valid question.")
    else:
        # Cap history to 20 entries (10 question-response pairs)
        if len(st.session_state.history) >= 20:
            st.session_state.history = st.session_state.history[-20:]  # Keep last 20
        progress_bar = st.progress(0)
        with st.spinner("Thinking..."):
            try:
                for i in range(1, 101):
                    progress_bar.progress(i / 100)
                    if i == 50:
                        logger.info(f"Processing query: {user_question}")
                        response_dict = process_query(user_question, st.session_state.history)
                        answer = response_dict["response"]
                        # Trim memory if nearing limit (simple length check)
                        if len(memory.buffer) > 10:  # Limit to 10 messages
                            memory.clear()
                            logger.info("Cleared memory due to length limit.")
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.history.append((user_question, answer, timestamp))
            except Exception as e:
                logger.error(f"Query processing failed: {str(e)}")
                st.error("Thinking a bit more...")  # Changed to positive message
                time.sleep(2)  # Brief pause
                try:
                    response_dict = process_query(user_question, st.session_state.history)
                    answer = response_dict["response"]
                except Exception as e2:
                    st.error(f"Retry failed: {str(e2)}. Please try again later.")
                    answer = f"Retry Error: {str(e2)}"
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.history.append((user_question, answer, timestamp))
            finally:
                progress_bar.empty()

# --- Render chat with new avatars ---
chat_html = '<div class="chat-container">'
for q, a, ts in st.session_state.history:
    chat_html += f'''
    <div class="chat-row user-row">
        <div style="width:40px;height:40px;border-radius:50%;background-color:#6B8E23;margin:0 10px;"></div>
        <div class="user-msg">{q}<div class="timestamp">{ts}</div></div>
    </div>
    <div class="chat-row">
        <img class="avatar" src="https://raw.githubusercontent.com/Lovely028/velana-aura-soma-guide-bot/main/bot_avatar.png" alt="Bot" onerror="this.style.display='none'; this.nextElementSibling.style.marginLeft='0';">
        <div class="bot-msg">{a}<div class="timestamp">{ts}</div></div>
    </div>
    '''
chat_html += '</div>'
st.markdown(chat_html, unsafe_allow_html=True)