

-----

# Velana's Aura-Soma Bot 🌈✨

Velana's Aura Soma Bot leverages a Retrieval-Augmented Generation (RAG) pipeline, Large Language Models (LLMs), Pinecone, and Streamlit to explore Aura-Soma® concepts. It combines AI, vector search, and generative reasoning to provide interactive guidance, bridging color-energy wisdom with cutting-edge technology.

## Table of Contents

Workflow Overview

Technology Stack

Project Structure

Core Components

aura_guide_bot_yt_transcription.py

aura_guide_bot_rag_llm_evaluator.py

app.py

Setup and Usage

Evaluation Pipeline

Streamlit Application

-----

## Workflow Overview

The project follows a comprehensive RAG pipeline from data ingestion to evaluation:

1.  **Data Ingestion & Transcription**: YouTube video content is automatically transcribed into text using OpenAI's **Whisper** model via the `aura_guide_bot_yt_transcription.py` script. This is supplemented by structured data from JSON files (`faq`, `pricelist`, `products`).
2.  **Data Processing & Indexing**: The raw text is cleaned, segmented into coherent chunks using `RecursiveCharacterTextSplitter`, and enriched with metadata.
3.  **Vector Embedding**: Each chunk is converted into a vector embedding using OpenAI's `text-embedding-3-large` model.
4.  **Indexing**: The embeddings and their corresponding text/metadata are upserted into a **Pinecone** serverless vector index for efficient similarity searching.
5.  **RAG Chain**: When a user submits a query through the Streamlit UI, the system:
      * Routes the query to the appropriate tool (e.g., product search, pricing lookup).
      * Retrieves the most relevant context chunks from Pinecone.
      * Combines the user query, retrieved context, and conversational history into a prompt.
      * Generates a context-aware response using a powerful LLM like **GPT-3.5-Turbo**.
6.  **Evaluation**: The bot's performance is rigorously tested using a dedicated evaluation pipeline in **LangSmith**, with **GPT-4.0** acting as an LLM-as-judge to score semantic accuracy.

-----

## Technology Stack

  * **LLMs & AI**: OpenAI (GPT-3.5-Turbo, GPT-4.0, Whisper, Embeddings)
  * **Framework**: LangChain
  * **Vector Database**: Pinecone
  * **Frontend**: Streamlit
  * **MLOps & Tracing**: LangSmith
  * **Data Extraction**: `yt-dlp` (for YouTube audio)

-----

## Project Structure

The project is organized to separate data processing, application logic, and configuration.

```
VELANA-AURA-SOMA-GUIDE-BOT/
├── .streamlit/
│   └── secrets.toml        # Streamlit secrets (API keys)
├── venv/                   # Python virtual environment
├── .gitignore              # Git ignore file
├── app.py                  # Main Streamlit application file
├── bot_avatar.png          # UI asset
├── chunks.json             # Output of processed text chunks
├── populate.py             # Script to process and upsert data to Pinecone
├── requirements.txt        # Python dependencies
└── data/                   # (Contains raw data files like JSONs and transcripts)
```

*(Note: Data files like `aura_soma_faq.json` and source codes are assumed to be in a `data/` or project files subdirectory or the root for clarity).*

-----

## Core Components

### `aura_guide_bot_yt_transcription.py`

This script automates the initial data collection phase from video content.

  * **Downloads** audio from specified YouTube URLs using `yt-dlp`.
  * **Transcribes** the audio to text with high accuracy using OpenAI's Whisper model.
  * **Saves** the transcripts as text files, preparing them for the RAG pipeline.

### `aura_guide_bot_rag_llm_evaluator.py`

This is the core of the project, containing the logic for the RAG pipeline and its evaluation.

  * **Data Processing**: Loads, cleans, and chunks data from all sources.
  * **Vector Store Initialization**: Connects to Pinecone, clears the namespace, and prepares it for ingestion.
  * **RAG Logic**: Defines the `VectorQueryTool` for retrieving context and the `process_query` function that orchestrates the entire retrieval and generation chain.
  * **Evaluation**: Contains the `evaluate_chatbot` function, which runs a test suite against a predefined dataset on LangSmith.

-----

## Setup and Usage

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd VELANA-AURA-SOMA-GUIDE-BOT
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Configure API Keys:**
    Create a `.streamlit/secrets.toml` file and add your API keys:
    ```toml
    OPENAI_API_KEY = "sk-..."
    PINECONE_API_KEY = "..."
    LANGCHAIN_API_KEY = "lsv2_..."
    ```
5.  **Populate the Vector Database:**
    Run the data processing and ingestion script.
    ```bash
    python populate.py
    ```
6.  **Run the Streamlit App:**
    ```bash
    streamlit run app.py
    ```

-----

## Evaluation Pipeline

The chatbot's reliability is ensured through a robust evaluation process managed by **LangSmith**.

  * **Dataset**: A curated dataset (`AuraGuideTest`) contains sample queries and their ideal "golden" responses.
  * **LLM-as-Judge**: The `evaluate_chatbot` function programmatically runs each test query through the bot. The generated response is compared against the golden response by a **GPT-4.0** model, which scores it for semantic accuracy.
  * **Hybrid Scoring**: The evaluation uses a hybrid approach, combining the GPT-4.0 semantic score with a keyword-matching score to ensure critical details (like booking links or product IDs) are present.
  * **Tracing**: All runs are traced in LangSmith, providing full visibility into the RAG chain's performance, latency, and token usage for debugging and optimization.

-----

## Streamlit Application

The user-facing component is a web application built with Streamlit, defined in `app.py`. It provides a clean, interactive chat interface where users can ask questions about Aura-Soma and receive instant, AI-driven responses. 
The app maintains conversational history to allow for context-aware follow-up questions.
