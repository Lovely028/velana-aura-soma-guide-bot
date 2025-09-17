import streamlit as st
import os
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
import re
from typing import List, Optional, Dict
import logging
import json

# --- Configure logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- API Key Security: Use Streamlit secrets ---
try:
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    INDEX_NAME = st.secrets["INDEX_NAME"]  # Ensure this is "aura-soma"
    CONFIG = {
        "booking_link": "https://velana.net/aurasoma#offers",
        "search_k": 5,
        "similarity_threshold": 0.1,
        "index_name": "aura-soma",
        "namespace": "aura-soma-velana",
        "embedding_dimensions": 1024,
        "embedding_model": "text-embedding-3-large"
    }
except KeyError as e:
    st.error(f"Missing secret key: {e}. Please set it in Streamlit secrets.")
    st.stop()

# --- Validate keys and index ---
if not PINECONE_API_KEY or not INDEX_NAME:
    st.error("PINECONE_API_KEY or INDEX_NAME is not set properly.")
    st.stop()

# --- Cache Pinecone initialization ---
@st.cache_resource
def initialize_pinecone():
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        if INDEX_NAME not in [idx["name"] for idx in pc.list_indexes().get("indexes", [])]:
            st.error(f"Index {INDEX_NAME} does not exist in us-east-1.")
            st.stop()
        index = pc.Index(INDEX_NAME)
        embeddings = OpenAIEmbeddings(
            model=CONFIG["embedding_model"],
            api_key=OPENAI_API_KEY,
            dimensions=CONFIG["embedding_dimensions"]
        )
        vector_store = PineconeVectorStore(
            index=index,
            embedding=embeddings,
            namespace=CONFIG["namespace"]
        )
        return vector_store
    except Exception as e:
        st.error(f"Pinecone initialization failed: {str(e)}")
        st.stop()

# --- Vector Query Tool ---
class VectorQueryTool:
    def __init__(self, vector_store, namespace, default_filter=None, post_process=None):
        self.vector_store = vector_store
        self.namespace = namespace
        self.default_filter = default_filter or {}
        self.post_process = post_process

    def query(self, query, extra_filter=None, fallback_to_general=True):
        filter_dict = {**self.default_filter, **(extra_filter or {})}
        try:
            results = self.vector_store.similarity_search_with_score(
                query=query.lower(),  # Normalize query
                k=CONFIG["search_k"],
                namespace=self.namespace,
                filter=filter_dict or None
            )
            filtered = [(doc.page_content, score) for doc, score in results if score > CONFIG["similarity_threshold"]]
            logger.info(f"Retrieved {len(filtered)} docs for '{query}' with scores: {[score for _, score in filtered]}")
            documents = [doc for doc, _ in filtered]
            if not documents and fallback_to_general and filter_dict:
                logger.info(f"No results with filter for '{query}'. Trying general search.")
                results = self.vector_store.similarity_search_with_score(
                    query=query.lower(),
                    k=CONFIG["search_k"],
                    namespace=self.namespace,
                    filter=None
                )
                filtered = [(doc.page_content, score) for doc, score in results if score > CONFIG["similarity_threshold"]]
                logger.info(f"General search retrieved {len(filtered)} docs with scores: {[score for _, score in filtered]}")
                documents = [doc for doc, _ in filtered]
            if self.post_process:
                documents = self.post_process(documents, query)
            return documents
        except Exception as e:
            logger.error(f"Vector store query error for '{query}': {str(e)}")
            return []

# --- Product Metadata Extraction ---
def build_product_pattern(id_prefix: str, num: int, category: str) -> re.Pattern:
    num_str = f"{num:03d}" if id_prefix == "B" else f"{num:02d}"
    return re.compile(rf'"{id_prefix}{num_str}".*?"name":"(.*?)".*?"category":"{category}".*?"description":"(.*?)"', re.IGNORECASE | re.DOTALL)

def extract_product_metadata(documents: List[str], query: str) -> List[str]:
    bottle_match = re.search(r'(?:B#?|bottle\s*#?|\#)(\d{1,3})', query, re.IGNORECASE)
    pomander_match = re.search(r'(?:P#?|pomander\s*#?|\#)(\d{1,2})', query, re.IGNORECASE)
    quintessence_match = re.search(r'(?:Q#?|quintessence\s*#?|\#)(\d{1,2})', query, re.IGNORECASE)

    if bottle_match:
        num = int(bottle_match.group(1))
        if 1 <= num <= 124:
            bottle_id = f"B{num:03d}"
            pattern = build_product_pattern("B", num, "Equilibrium")
            for doc in documents:
                match = pattern.search(doc)
                if match:
                    name, desc = match.groups()
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
                    return [f"Quintessence {quintessence_id} is {name}. Description: {desc}."]
    return documents

def append_booking_redirect(documents: List[str], query: str) -> List[str]:
    return documents + [f"To book a consultation, visit: {CONFIG['booking_link']}"]

# --- Initialize Tools ---
vector_store = initialize_pinecone()
tools = {
    "faq": VectorQueryTool(vector_store, CONFIG["namespace"], {"topic": {"$in": ["general", "services", "pricing", "shipping", "authenticity_sustainability", "discounts", "returns"]}}),
    "pricing": VectorQueryTool(vector_store, CONFIG["namespace"], {"topic": "pricing"}),
    "product": VectorQueryTool(vector_store, CONFIG["namespace"], {"topic": "products"}, post_process=extract_product_metadata),
    "meet_aura_soma": VectorQueryTool(vector_store, CONFIG["namespace"], {"document_type": "transcript"}),
    "booking": VectorQueryTool(vector_store, CONFIG["namespace"], {"topic": "pricing"}, post_process=append_booking_redirect)
}

# --- Initialize LLM and Memory ---
memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, api_key=OPENAI_API_KEY)

# --- System Prompt ---
system_prompt = """You are AuraGuide, a helpful, mindful chatbot for Velana.net, specializing in Aura-Soma—a holistic color-care system for wellbeing. Respond concisely, warmly, using language inspired by light and harmony. Be positive, non-judgemental, and emphasize personal resonance.

Key Rules:
- Use provided {context} if relevant; otherwise, use general knowledge with caution.
- For general Aura-Soma questions, including shipping and returns, use the context and end with: 'To dive deeper, book a consultation at {booking_link}.'
- For bottle/pomander/quintessence queries, use the context to extract id, name, category, and description (e.g., "B001 - Blue/Deep Magenta - Physical Rescue"). Then say: 'I'm not a practitioner, so I can't provide full analysis. Book with Velana: {booking_link}.'
- For pricing/services/consultation questions, use the context for name, price, description. Note: Bottles ideally follow consultation (47€), Pomanders/Quintessences are 27€ each.
- If no relevant info, say: 'I couldn’t find specific details, but generally [brief info]. Book at {booking_link}.'
- Always redirect to {booking_link} for personalized advice.
- Maintain conversational memory."""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "Chat History: {chat_history}\nContext: {context}\nQuery: {query}")
])

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

# --- Session state (no JSON file) ---
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
    memory.clear()  # Clear memory too
    st.rerun()

# --- Cache chunks.json load ---
@st.cache_data
def load_chunks():
    try:
        with open("chunks.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("chunks.json not found. Please add it to the repo root.")
        st.stop()
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON in chunks.json: {str(e)}")
        st.stop()

# --- Local JSON Search Fallback ---
def local_json_search(query: str, chunks: List[dict]) -> List[str]:
    matches = []
    query_lower = query.lower()
    for chunk in chunks:
        if query_lower in chunk["content"].lower() or query_lower in str(chunk["metadata"]).lower():
            matches.append(chunk["content"])
    # Specific regex for products (e.g., B15 matches b015)
    bottle_match = re.search(r'(?:b#?|bottle\s*#?|\#)(\d{1,3})', query_lower, re.IGNORECASE)
    if bottle_match:
        num = int(bottle_match.group(1))
        bottle_id = f"b{num:03d}"
        for chunk in chunks:
            if bottle_id in chunk["content"].lower() or bottle_id in str(chunk["metadata"]).lower():
                matches.append(chunk["content"])
    # Specific for booking
    if "book" in query_lower or "consultation" in query_lower:
        for chunk in chunks:
            if "book" in chunk["content"].lower() or "consultation" in chunk["content"].lower():
                matches.append(chunk["content"])
    return matches

# --- User input with validation ---
user_question = st.text_input("Type your question and press Enter:")
if user_question:
    user_question = user_question.strip()[:500].lower()
    if not user_question:
        st.warning("Please enter a valid question.")
    else:
        progress_bar = st.progress(0)
        with st.spinner("Thinking..."):
            try:
                for i in range(1, 101):
                    progress_bar.progress(i / 100)
                    if i == 50:
                        route = route_query(user_question)
                        tool = tools.get(route, tools["faq"])
                        context = tool.query(user_question)
                        if len(context) < 3:  # Weak Pinecone retrieval
                            chunks = load_chunks()
                            fallback_matches = local_json_search(user_question, chunks)
                            context += fallback_matches
                            st.info("Using chunks.json fallback for better results.")
                        chat_history = memory.load_memory_variables({})["chat_history"]
                        response = (prompt_template | llm).invoke({
                            "query": user_question,
                            "context": "\n---\n".join(context) if context else "No context available",
                            "chat_history": chat_history,
                            "booking_link": CONFIG["booking_link"]
                        }).content
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.session_state.history.append((user_question, response, timestamp))
                        memory.save_context({"input": user_question}, {"output": response})
            except Exception as e:
                st.error(f"Query failed: {str(e)}. Please check your internet or API keys.")
            finally:
                progress_bar.empty()

# --- Render chat with new avatars ---
chat_html = '<div class="chat-container">'
for q, a, ts in st.session_state.history:
    # User message with midtone green circle avatar
    chat_html += f'''
    <div class="chat-row user-row">
        <div style="width:40px;height:40px;border-radius:50%;background-color:#6B8E23;margin:0 10px;"></div>
        <div class="user-msg">{q}<div class="timestamp">{ts}</div></div>
    </div>
    <div class="chat-row">
        <!-- Bot avatar: Turquoise sparkling neon digital art from GitHub main -->
        <img class="avatar" src="https://raw.githubusercontent.com/Lovely028/velana-aura-soma-guide-bot/main/bot_avatar.png" alt="Bot" onerror="this.style.display='none'; this.nextElementSibling.style.marginLeft='0';">
        <div class="bot-msg">{a}<div class="timestamp">{ts}</div></div>
    </div>
    '''
chat_html += '</div>'
st.markdown(chat_html, unsafe_allow_html=True)

# --- Helper Functions ---
def route_query(query: str) -> str:
    routing_prompt = ChatPromptTemplate.from_template(
        "Classify into one: faq, pricing, product, meet_aura_soma, booking, general. Query: {query}"
    )
    try:
        route = llm.invoke(routing_prompt.format(query=query)).content.strip().lower()
        return route
    except Exception as e:
        logger.error(f"Error routing query '{query}': {str(e)}")
        return "faq"