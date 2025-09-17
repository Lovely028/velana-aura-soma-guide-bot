import streamlit as st
import os
from datetime import datetime
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
import json
import re
from typing import List, Dict
import logging
import time

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
        return OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1024)
    except Exception as e:
        st.error(f"OpenAI Embeddings failed: {str(e)}")
        st.stop()

# --- Cache vector store ---
@st.cache_resource
def get_vector_store(_embeddings):
    try:
        vector_store = PineconeVectorStore(
            index_name=INDEX_NAME,
            embedding=_embeddings,
            namespace="aura-soma-velana"
        )
        return vector_store.as_retriever(search_kwargs={"k": 5, "score_threshold": 0.1})
    except Exception as e:
        st.error(f"Pinecone vector store loading failed: {str(e)}")
        st.stop()

# --- Cache LLM ---
@st.cache_resource
def get_llm(model_name="gpt-3.5-turbo", temperature=0):
    try:
        return ChatOpenAI(model_name=model_name, temperature=temperature)
    except Exception as e:
        st.error(f"OpenAI LLM initialization failed: {str(e)}")
        st.stop()

# --- Cache RAG chain ---
@st.cache_resource
def get_qa_chain(_llm, _retriever):
    return RetrievalQA.from_chain_type(llm=_llm, retriever=_retriever)

# Initialize cached resources
pinecone_instance = init_pinecone()
embeddings = get_embeddings()
retriever = get_vector_store(embeddings)
llm = get_llm()
qa_chain = get_qa_chain(llm, retriever)

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

# --- Local JSON Search Fallback ---
def local_json_search(query: str, chunks: List[Dict]) -> List[str]:
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
    return matches if matches else ["No specific info found in fallback."]

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
    model_choice = st.selectbox("Choose model:", ["gpt-3.5-turbo", "gpt-4"])
    temp = st.slider("Temperature:", 0.0, 1.0, 0.0)
    llm.model_name = model_choice
    llm.temperature = temp
    qa_chain = get_qa_chain(llm, retriever)

# --- Clear chat button ---
if st.button("🗑️ Clear Chat"):
    st.session_state.history = []
    st.rerun()  # Use st.rerun() instead of experimental_rerun() for newer Streamlit versions

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
        with st.spinner("Looking for the Best Answer for You..."):  # Updated spinner message
            try:
                for i in range(1, 101):
                    progress_bar.progress(i / 100)
                    if i == 50:
                        # Try Pinecone retrieval first
                        docs = retriever.get_relevant_documents(user_question)
                        context = [doc.page_content for doc in docs]
                        if len(context) < 3:  # Weak retrieval - fallback to JSON
                            fallback_matches = local_json_search(user_question, chunks)
                            context += fallback_matches
                        # Removed st.info, keeping the spinner message as user feedback
                        answer = qa_chain.run("\n---\n".join(context) if context else user_question)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.history.append((user_question, answer, timestamp))
            except Exception as e:
                logger.error(f"Query processing failed: {str(e)}")
                st.error(f"Query failed: {str(e)}. Please check your internet or API keys or try again later.")
                st.session_state.history.append((user_question, f"Error: {str(e)}", timestamp))
                time.sleep(2)  # Brief pause before retry
                try:
                    docs = retriever.get_relevant_documents(user_question)
                    context = [doc.page_content for doc in docs]
                    if len(context) < 3:
                        fallback_matches = local_json_search(user_question, chunks)
                        context += fallback_matches
                    answer = qa_chain.run("\n---\n".join(context) if context else user_question)
                    st.session_state.history.append((user_question, answer, timestamp))
                except Exception as e2:
                    st.error(f"Retry failed: {str(e2)}. Please try again later.")
                    st.session_state.history.append((user_question, f"Retry Error: {str(e2)}", timestamp))
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