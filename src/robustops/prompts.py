"""Editable LLM-assistance prompts in the repository's prompts directory."""
from pathlib import Path


PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts"


def read_prompt(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8")
