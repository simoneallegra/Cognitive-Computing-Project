"""Punteggio euristico (solo regole, niente modello) della qualita' di
una risorsa trovata su internet."""
from __future__ import annotations

from urllib.parse import urlparse

from langchain_core.tools import tool

_RECENCY_KEYWORDS = ("2024", "2025", "2026", "this year", "latest", "new", "upcoming")
_LOW_TRUST_DOMAINS = ("pinterest.", "quora.", "reddit.com")
_INTERESTINGNESS_MARKERS = (
    "unique", "hidden gem", "surprising", "lesser-known", "secret",
    "must-see", "award", "record", "first-ever", "exclusive",
)


def score_resource(text: str, url: str = "") -> tuple[float, list[str]]:
    score = 0.5
    reasons: list[str] = []
    text_lower = text.lower()

    word_count = len(text.split())
    if word_count >= 150:
        score += 0.15
        reasons.append("substantial content length")
    elif word_count < 40:
        score -= 0.2
        reasons.append("very short / snippet-only content")

    if any(k in text_lower for k in _RECENCY_KEYWORDS):
        score += 0.15
        reasons.append("contains recency signals")

    domain = urlparse(url).netloc if url else ""
    if any(d in domain for d in _LOW_TRUST_DOMAINS):
        score -= 0.25
        reasons.append(f"low-trust domain ({domain})")
    elif domain:
        score += 0.05
        reasons.append(f"identifiable source domain ({domain})")

    if any(m in text_lower for m in _INTERESTINGNESS_MARKERS):
        score += 0.1
        reasons.append("contains 'interestingness' markers")

    return max(0.0, min(1.0, score)), reasons


@tool
def resource_quality_tool(text: str, url: str = "") -> str:
    """Score a retrieved resource for quality/interestingness (0-1), with
    a short justification. Use before drafting to prefer good sources."""
    score, reasons = score_resource(text, url)
    reason_text = "; ".join(reasons) if reasons else "no strong signals"
    return f"quality_score={score:.2f}; reasons={reason_text}"


if __name__ == "__main__":
    # Debug: `python tools/quality_tool.py` prova l'euristica su tre casi
    # rappresentativi: fonte solida, snippet corto, dominio poco affidabile
    samples = [
        (
            "This hidden gem destination, discovered in 2025, offers a substantial "
            "amount of unique cultural experiences rarely covered elsewhere. " * 5,
            "https://example-travel-guide.com/article",
        ),
        ("Short snippet.", ""),
        ("Some generic content about a place.", "https://pinterest.com/pin/123"),
    ]
    for text, url in samples:
        score, reasons = score_resource(text, url)
        print(f"score={score:.2f} url={url!r} reasons={reasons}")
