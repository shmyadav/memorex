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

## Running with Docker (Recommended)

The easiest way to run MemoRex along with a Neo4j database is using Docker Compose. This resolves any local connection issues and ensures a consistent environment.

### 1. Build and Start

```bash
# Start Neo4j and the MemoRex application
docker-compose up --build
```

### 2. Access Neo4j Browser

You can access the Neo4j management UI at:
[http://localhost:7474](http://localhost:7474)

- **Username**: `neo4j`
- **Password**: `password` (default)

### 3. Note on Hostnames

When running the application inside Docker, set your **NEO4J_URI** in `.env` to:
`bolt://neo4j:7687`

The `app` container uses the service name `neo4j` to communicate with the database.

## Running Locally

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
