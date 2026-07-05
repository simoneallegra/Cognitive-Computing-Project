"""Debug: `python -m tools` elenca i tool assemblati in REACT_TOOLS e il
loro schema di argomenti, cosi' come li vedra' il LLM dopo .bind_tools(...)."""
from __future__ import annotations

from tools import REACT_TOOLS

for t in REACT_TOOLS:
    print(f"- {t.name}: {t.args}")
