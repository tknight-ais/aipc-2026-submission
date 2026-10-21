"""Loads prompt templates from src/aipc/prompts/ - the one place prompt
text lives, so it can be versioned and hashed rather than buried in code.
"""

import hashlib
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def load_prompt(name: str) -> tuple[str, str]:
    """Returns (text, sha256_hash) for prompts/<name>.md."""
    text = (PROMPTS_DIR / f"{name}.md").read_text()
    prompt_hash = hashlib.sha256(text.encode()).hexdigest()
    return text, prompt_hash

