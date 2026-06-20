"""
src/config.py
Central configuration — loads from .env file.
All other modules import from here; never read os.environ directly.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (two levels up from this file)
_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env", override=False)

# ── MongoDB ───────────────────────────────────────────────────────────────────
MONGO_URI: str     = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME: str = os.getenv("MONGO_DB_NAME", "insurance_rag")
MONGO_TIMEOUT: int = int(os.getenv("MONGO_TIMEOUT", "30"))

# ── Ollama ────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL: str       = os.getenv("LLM_MODEL", "llama3.2:3b")
EMBED_MODEL: str     = os.getenv("EMBED_MODEL", "nomic-embed-text")

# ── Embedding ─────────────────────────────────────────────────────────────────
EMBED_DIMENSION: int = 768          # nomic-embed-text output dimension
VECTOR_INDEX_NAME: str = "vector_index"

# ── Retrieval ─────────────────────────────────────────────────────────────────
DEFAULT_TOP_K: int        = 8       # chunks returned per agent
CLAIMS_TOP_K: int         = 5       # historical claims returned
NUM_CANDIDATES_MULT: int  = 10      # numCandidates = top_k * this
RRF_K: int                = 60      # Reciprocal Rank Fusion k constant

# ── Agent workflow ────────────────────────────────────────────────────────────
MAX_RETRIES: int        = 3
CONFIDENCE_THRESHOLD: float = 0.7   # below → retry or fallback
FALLBACK_THRESHOLD: float   = 0.4   # below → skip retry, go fallback

# ── Session management ────────────────────────────────────────────────────────
SESSION_TTL_HOURS: int           = int(os.getenv("SESSION_TTL_HOURS", "24"))
SUMMARY_EVERY_N: int             = int(os.getenv("SUMMARY_EVERY_N_MESSAGES", "5"))
MAX_MESSAGES_PER_SESSION: int    = int(os.getenv("MAX_MESSAGES_PER_SESSION", "100"))
LANGCHAIN_MEMORY_K: int          = 5    # ConversationBufferWindowMemory window

# ── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE_POLICY: int     = 512
CHUNK_OVERLAP_POLICY: int  = 64
CHUNK_SIZE_REG: int        = 400
CHUNK_OVERLAP_REG: int     = 50
CHUNK_SIZE_CLAIMS: int     = 300    # one claim per chunk, no overlap

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_RAW_DIR: Path       = _ROOT / "data" / "raw"
DATA_SYNTHETIC_DIR: Path = _ROOT / "data" / "synthetic"
