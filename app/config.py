import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5-5")
# Multilingual model so Hebrew and English course material both embed well.
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))

CHUNK_SIZE = 1000  # characters
CHUNK_OVERLAP = 150
TOP_K = 5
# If everything being searched fits under this many characters (~15-20K tokens),
# send all of it instead of the top-k chunks. Broad questions like "summarize this
# file" need the whole document, which similarity search can't provide.
FULL_CONTEXT_CHARS = 60_000
