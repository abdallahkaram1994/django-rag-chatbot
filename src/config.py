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

# Rate limiting (protects the free Gemini quota).
RATE_PER_MINUTE = int(os.getenv("RATE_PER_MINUTE", "10"))  # per client IP
DAILY_CAP = int(os.getenv("DAILY_CAP", "300"))  # all clients combined
