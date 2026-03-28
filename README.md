# MemoRex

MemoRex is a graph-based memory and knowledge management system designed to process episodes of information into a structured Neo4j graph with LLM-powered nodes and edges.

## Prerequisites

- **Python 3.10+**
- **Neo4j Desktop or AuraDB Instance** (See [Neo4j Setup](https://neo4j.com/download/))
- **OpenAI API Key** (Required for LLM and embeddings)

## Quick Start

### 1. Set Up Environment

We recommend using a virtual environment to manage dependencies:

```bash
# Create a virtual environment
python3 -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Alternatively, you can use the provided setup script:
```bash
chmod +x setup.sh
./setup.sh
```

### 2. Configure Environment Variables

Copy the example environment file and fill in your credentials:

```bash
cp .env.example .env
```

Edit the `.env` file with your **OPENAI_API_KEY** and **Neo4j** connection details.

### 3. Run MemoRex

You can run the example simulation in `main.py`:

```bash
python3 main.py
```

## Core Functionality

- **Node/Edge Extraction**: Automatically identifies entities and relationships from text.
- **Episodic Management**: Tracks information across time with episodic nodes.
- **LLM/Embedder Integration**: Seamless connection to OpenAI for semantic analysis and ranking.
- **Neo4j Storage**: Fast, scalable graph storage with full-text and vector search indices.

## Directory Structure

- `main.py`: Main entry point and orchestration.
- `llm.py`: LLM client implementation.
- `embedder.py`: OpenAI embedding services.
- `graph_driver.py`: Neo4j driver and query execution.
- `datamodels.py`: Pydantic models for nodes and edges.
- `search_utils.py`: Powerful graph search and retrieval utilities.
