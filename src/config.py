import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

CHROMA_PATH = str(ROOT / os.getenv("CHROMA_DIR", "chroma"))

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
# Where fastembed keeps the downloaded model. Without this it defaults to a temp
# folder that can be wiped between restarts.
MODEL_CACHE = os.getenv("FASTEMBED_CACHE_PATH", str(ROOT / "models"))

TOP_K = 5
# Best-match distance above which we refuse without calling the LLM.
# Provisional: answerable questions peaked near 0.52, off-topic ones started near 0.65.
# Recalibrate after rebuilding the index with fastembed.
MAX_DISTANCE = float(os.getenv("MAX_DISTANCE", "0.6"))

# Copy the model name from Google AI Studio into .env (GEMINI_MODEL=...).
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "")

# Per-visitor limits (by client IP).
RATE_PER_MINUTE = int(os.getenv("RATE_PER_MINUTE", "10"))
IP_DAILY_CAP = int(os.getenv("IP_DAILY_CAP", "40"))

# Shared Gemini budget, counted in Gemini calls across all visitors. Keep both a bit under
# the free-tier limits shown in AI Studio (gemini-3.5-flash-lite: 500/day, 15/min), so the
# owner's own testing still fits. The quota day resets at midnight Pacific time.
GEMINI_DAILY_BUDGET = int(os.getenv("GEMINI_DAILY_BUDGET", "400"))
GEMINI_PER_MINUTE = int(os.getenv("GEMINI_PER_MINUTE", "12"))
QUOTA_TIMEZONE = "America/Los_Angeles"

# Follow-up questions. "llm": Gemini rewrites a follow-up into a standalone question (one extra call,
# only when there is chat history). "none": search with the question as typed.
FOLLOWUP_MODE = os.getenv("FOLLOWUP_MODE", "llm")
HISTORY_TURNS = int(os.getenv("HISTORY_TURNS", "3"))  # exchanges (user + assistant) kept

CACHE_SIZE = int(os.getenv("CACHE_SIZE", "500"))
CACHE_SEED = ROOT / "cache" / "seed.json"
