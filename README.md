# Insurance Claim Agentic RAG System

An intelligent, locally-running agentic RAG (Retrieval-Augmented Generation) system that helps hospital insurance departments analyze patient cases against insurance policies, historical claims, and regulatory guidelines.

## Overview

Hospital insurance staff spend hours manually cross-referencing patient cases against policy documents, provider agreements, historical claims, and IRDA guidelines. This system automates that process using a locally-running agentic RAG pipeline:

1. **User** describes a patient case in plain English
2. **Agent** intelligently routes the query to relevant data sources
3. **System** retrieves policy clauses, similar claims, and regulations
4. **AI** generates structured recommendations with citations and reasoning

## Features

- **100% Local** - No cloud APIs, no data leaves your machine
- **Agentic Workflow** - Multi-agent pipeline with routing, retrieval, reflection, and synthesis
- **Hybrid Retrieval** - Combines semantic vector search with metadata filtering
- **Multi-Turn Conversations** - Maintains context across queries
- **Human-in-the-Loop** - Clarification questions and escalation for uncertain cases
- **Audit Logging** - Complete query history for compliance
- **Synthetic Data** - Realistic insurance data for testing without PHI concerns

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Streamlit UI                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐               │
│  │  Query   │  │Document  │  │  Audit   │               │
│  │ Interface│  │  Upload  │  │   Log    │               │
│  └──────────┘  └──────────┘  └──────────┘               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 LangGraph Agent Workflow                   │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐    │
│  │ Query   │→ │Parallel │→ │Reflect- │→ │ Answer  │    │
│  │ Router  │  │Retrieval│  │  ion    │  │Synthesis│    │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              MongoDB (Vector + Document Store)             │
│  ┌─────────────┐  ┌───────────┐  ┌───────────────┐      │
│  │document_    │  │historical_│  │conversation_  │      │
│  │chunks       │  │claims     │  │history        │      │
│  │(w/ vectors) │  │           │  │               │      │
│  └─────────────┘  └───────────┘  └───────────────┘      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Ollama (Local LLMs)                     │
│         ┌─────────────────┐  ┌─────────────────┐         │
│         │  llama3.2:3b    │  │ nomic-embed-text│         │
│         │  (Chat/Reason)  │  │  (Embeddings)   │         │
│         └─────────────────┘  └─────────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **LLM** | Ollama + llama3.2:3b | Chat, reasoning, intent classification |
| **Embeddings** | Ollama + nomic-embed-text | Vector generation for retrieval |
| **Vector Database** | MongoDB + $vectorSearch | Store and search document embeddings |
| **Agent Framework** | LangGraph | Multi-agent workflow orchestration |
| **UI** | Streamlit | Interactive dashboard |
| **Data Generation** | Faker | Synthetic insurance data |

## Project Structure

```
insurance/
├── data/
│   ├── raw/                 # Source documents (policies, claims, regulations)
│   └── synthetic/           # Generated synthetic data
├── scripts/
│   ├── generate_synthetic_data.py
│   ├── setup_mongo.py
│   ├── ingest_documents.py
│   └── verify_setup.py
├── src/
│   ├── agents/              # LangGraph agent definitions
│   │   ├── state.py
│   │   ├── query_router.py
│   │   ├── retrieval_agent.py
│   │   ├── reflection_agent.py
│   │   ├── answer_agent.py
│   │   └── graph.py
│   ├── database/            # MongoDB connection & vector store
│   ├── ingestion/           # Document loading & chunking
│   ├── retrieval/           # Semantic + hybrid search
│   ├── memory/              # Conversation memory
│   ├── api/                 # Query/Response models
│   └── audit/               # Audit logging
├── ui/
│   ├── app.py               # Streamlit main entry
│   └── pages/               # Query, Documents, Audit Log
├── tests/
│   └── sample_queries.json
├── requirements.txt
└── README.md
```

## Quick Start

### Prerequisites

- **Python 3.11+**
- **Ollama** (for local LLMs)
- **MongoDB 6.0+** (with vector search support)
- **8GB+ RAM** recommended

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/thanmayee-shetty/insurance_claim.git
cd insurance_claim
```

**2. Install Python dependencies**
```bash
pip install -r requirements.txt
```

**3. Install and configure Ollama**
```bash
# Install Ollama from https://ollama.com
# Pull required models
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

**4. Install and configure MongoDB**

Download MongoDB Community Edition from [mongodb.com](https://www.mongodb.com/try/download/community) or use Docker:
```bash
docker run -d -p 27017:27017 --name mongodb mongo:latest
```

**5. Configure environment**
```bash
copy .env.example .env
# Edit .env with your MongoDB URI
MONGO_URI=mongodb://localhost:27017
MONGO_DB_NAME=insurance_rag
```

**6. Setup database and generate data**
```bash
python scripts/setup_mongo.py
python scripts/generate_synthetic_data.py
python scripts/ingest_documents.py
```

**7. Run the application**
```bash
streamlit run ui/app.py
```

Open http://localhost:8501 in your browser.

## Agent Workflow

The system uses a 4-stage agentic pipeline:

### 1. Query Routing Agent
- Analyzes query intent (coverage check, claims precedent, regulatory compliance)
- Extracts entities (policy numbers, diagnoses, procedures)
- Routes to appropriate data sources

### 2. Parallel Retrieval Agents
- Multiple sub-agents search policies, claims, regulations, and agreements simultaneously
- Each uses hybrid search (vector similarity + metadata filters)
- Results merged using Reciprocal Rank Fusion

### 3. Reflection Agent
- Validates retrieved chunks against the query
- Scores confidence (0.0-1.0)
- Decides: Answer, Retry with expanded search, or Fallback

### 4. Answer Synthesis Agent
- Generates structured response with citations
- Adds recommendation (APPROVE, REJECT, ESCALATE, NEEDS_MORE_INFO)
- Shows reasoning chain for transparency

### Human-in-the-Loop
- Clarification questions for ambiguous queries
- Escalation for low-confidence results
- Audit log for compliance

## Sample Queries

| Query | Expected Behavior |
|-------|-------------------|
| "Does the Star Health policy cover knee replacement for a 62-year-old?" | Search policies + claims, assess coverage |
| "Show me claims where cardiac bypass was rejected" | Search historical claims, filter by outcome |
| "What does IRDA say about pre-authorisation timelines?" | Search regulatory documents |
| "Is dialysis covered for a diabetic patient on a 2-year policy?" | Hybrid search (policies + claims + regulations) |

## Security & Compliance

-  **No real PHI** - All data is synthetic
-  **100% Local** - No cloud API calls
-  **Audit Trail** - Every query logged immutably
-  **No Data Leakage** - All processing on local machine

## Performance

| Component | Specification |
|-----------|---------------|
| **Total Documents** | 35 (policies, claims, regulations, agreements) |
| **Total Chunks** | ~370 |
| **Vector Storage** | ~1.1 MB |
| **Embedding Model** | nomic-embed-text (768-dim) |
| **LLM** | llama3.2:3b (7GB RAM, 1.2s latency) |
| **Full Query Pipeline** | ~60-120 seconds |

## Contributing

This is a portfolio project. Suggestions and improvements welcome!

## License

MIT License - feel free to use for learning and portfolio purposes.

## Acknowledgments

- LangChain & LangGraph for agent orchestration
- Ollama for local LLMs
- MongoDB for vector search
- Streamlit for the UI
