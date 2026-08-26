"""Generatore di post "classifica": data una lista di destinazioni e una
categoria, produce un "Top N citta' per {categoria}"

A differenza degli altri due generatori (una destinazione sola), questo
prende una lista intera, stesso schema campi degli altri. Spareggio:
vince chi ha la media piu' alta sulle altre 8 categorie, poi l'ordine
originale della lista (sort stabile)
"""

from __future__ import annotations

VALID_CATEGORIES = (
    "culture", "adventure", "nature", "beaches",
    "nightlife", "cuisine", "wellness", "urban", "seclusion",
)

# nomi leggibili delle categorie, per titolo e headline
CATEGORY_TITLE = {
    "culture": "Culture",
    "adventure": "Adventure",
    "nature": "Nature",
    "beaches": "Beaches",
    "nightlife": "Nightlife",
    "cuisine": "Food & Cuisine",
    "wellness": "Wellness",
    "urban": "City Life",
    "seclusion": "Peace & Seclusion",
}

# frase breve sotto il punteggio di ogni citta', per categoria e voto
RANK_BLURBS = {
    "culture": {
        1: "light on cultural sights, but easy to skip without missing much",
        2: "a few cultural gems if you go looking",
        3: "a solid dose of museums, history and local tradition",
        4: "deep cultural roots — history and art around every corner",
        5: "an absolute cultural powerhouse, top of the list for a reason",
    },
    "adventure": {
        1: "not the spot for adrenaline seekers",
        2: "a little outdoor action if you make time for it",
        3: "decent options for hiking and outdoor activities",
        4: "a genuine playground for adventure lovers",
        5: "as good as it gets for adventure travel",
    },
    "nature": {
        1: "don't come here for the scenery",
        2: "modest natural surroundings",
        3: "pleasant landscapes worth a half-day detour",
        4: "stunning natural surroundings, well worth the trip alone",
        5: "jaw-dropping nature that's reason enough to book the flight",
    },
    "beaches": {
        1: "skip the swimsuit, beaches aren't the draw here",
        2: "beaches exist, but they're not why you'd come",
        3: "solid coastline for a relaxed day out",
        4: "gorgeous beaches that are hard to beat",
        5: "postcard-perfect beaches, among the best anywhere",
    },
    "nightlife": {
        1: "early nights are the norm here",
        2: "a quiet evening scene, nothing major",
        3: "a fun night out if you know where to look",
        4: "a lively after-dark scene with real staying power",
        5: "legendary nightlife — the parties barely stop",
    },
    "cuisine": {
        1: "food is functional, not a highlight",
        2: "a few good meals if you pick carefully",
        3: "tasty, reliable local cuisine",
        4: "a food scene that's reason enough to visit",
        5: "world-class cuisine that steals the whole trip",
    },
    "wellness": {
        1: "not built for relaxation or spa time",
        2: "limited wellness options",
        3: "decent spa and relaxation opportunities",
        4: "genuinely rejuvenating, with strong wellness options",
        5: "a true wellness sanctuary",
    },
    "urban": {
        1: "rural and low-key, far from any city buzz",
        2: "quiet and low-density, minimal city energy",
        3: "a fair mix of city buzz and quieter corners",
        4: "busy, metropolitan energy throughout",
        5: "full-blown metropolis energy, fast-paced and dense",
    },
    "seclusion": {
        1: "expect crowds pretty much everywhere",
        2: "quiet spots exist, but you'll have to hunt for them",
        3: "fairly easy to find a quiet corner away from the crowds",
        4: "real secluded spots worth seeking out",
        5: "true peace and quiet, about as secluded as it gets",
    },
}


def _clamp_score(value) -> int:
    """Forza il punteggio nel range 1-5"""
    try:
        value = int(value)
    except (TypeError, ValueError):
        return 3
    return max(1, min(5, value))


def _other_categories_average(entry, category: str) -> float:
    """Media delle altre 8 categorie, per lo spareggio"""
    others = []
    for c in VALID_CATEGORIES:
        if c != category:
            others.append(c)

    scores = []
    for c in others:
        scores.append(_clamp_score(entry.get(c, 3)))

    return sum(scores) / len(scores) if scores else 0.0


def _location_text(entry) -> str:
    """Costruisce la stringa citta'/paese, con regione se presente"""
    city = entry.get("city", "Unknown")
    region = entry.get("region", "")
    country = entry.get("country", "")
    if region and region.lower() != city.lower():
        return f"{city}, {region}, {country}"
    return f"{city}, {country}"


def rank_destinations(destinations, category: str, top_n: int = 5):
    """Ordina le destinazioni per punteggio nella categoria, spareggio con
    la media delle altre 8 categorie, poi ordine originale della lista"""
    if category not in VALID_CATEGORIES:
        raise ValueError(
            f"Unknown category '{category}'. Must be one of: {', '.join(VALID_CATEGORIES)}"
        )

    # sort stabile: a parita' di punteggio resta l'ordine originale
    enriched = []
    for entry in destinations:
        rank_score = _clamp_score(entry.get(category, 3))
        tiebreak_score = _other_categories_average(entry, category)
        enriched.append({**entry, "_rank_score": rank_score, "_tiebreak_score": tiebreak_score})

    ranked = sorted(enriched, key=lambda e: (e["_rank_score"], e["_tiebreak_score"]), reverse=True)
    return ranked[:top_n]


def generate_ranking_post(destinations, category: str, top_n: int = 5) -> str:
    """Genera il post "Top N citta' per {categoria}" """
    category = category.lower().strip()
    top_destinations = rank_destinations(destinations, category, top_n)

    if not top_destinations:
        return f"No destinations available to rank for {CATEGORY_TITLE.get(category, category)}."

    title_category = CATEGORY_TITLE.get(category, category.capitalize())
    actual_n = len(top_destinations)

    intro = (
        f"Looking for the best places to experience {title_category.lower()}? "
        f"Here are the top {actual_n} destinations that deliver, ranked by how well they score on it."
    )

    entries_text = []

    for i, entry in enumerate(top_destinations, start=1):
        city = entry.get("city", "Unknown")
        location = _location_text(entry)
        score = entry["_rank_score"]
        budget = entry.get("budget_level", "")
        blurb = RANK_BLURBS.get(category, {}).get(score, "")

        block = f"{i}. **{city}** — {location}\nScore: {score}/5 — {blurb.capitalize()}."
        if budget:
            block += f' Budget level: "{budget}".'

        entries_text.append(block)

    entries_block = "\n\n".join(entries_text)

    winner_city = top_destinations[0].get("city", "this destination")
    closing = (
        f"If {title_category.lower()} is your priority, {winner_city} is the clear pick — "
        f"but any of these {actual_n} are worth shortlisting depending on your dates and budget."
    )

    post = f"""Top {actual_n} Cities for {title_category}

{intro}

---

{entries_block}

---

{closing}"""

    return post.strip()


# esempio d'uso
if __name__ == "__main__":
    sample_destinations = [
        {
            "city": "Lisbon", "country": "Portugal", "region": "Lisbon",
            "budget_level": "medium",
            "culture": 4, "adventure": 3, "nature": 3, "beaches": 4,
            "nightlife": 5, "cuisine": 5, "wellness": 3, "urban": 4, "seclusion": 2,
        },
        {
            "city": "Berlin", "country": "Germany", "region": "Berlin",
            "budget_level": "medium",
            "culture": 4, "adventure": 2, "nature": 2, "beaches": 1,
            "nightlife": 5, "cuisine": 3, "wellness": 2, "urban": 5, "seclusion": 1,
        },
        {
            "city": "Ibiza", "country": "Spain", "region": "Balearic Islands",
            "budget_level": "high",
            "culture": 2, "adventure": 3, "nature": 3, "beaches": 5,
            "nightlife": 5, "cuisine": 3, "wellness": 4, "urban": 2, "seclusion": 2,
        },
        {
            "city": "Bangkok", "country": "Thailand", "region": "Bangkok",
            "budget_level": "low",
            "culture": 4, "adventure": 3, "nature": 2, "beaches": 1,
            "nightlife": 5, "cuisine": 5, "wellness": 3, "urban": 5, "seclusion": 1,
        },
        {
            "city": "Reykjavik", "country": "Iceland", "region": "Capital Region",
            "budget_level": "high",
            "culture": 2, "adventure": 5, "nature": 5, "beaches": 1,
            "nightlife": 2, "cuisine": 3, "wellness": 4, "urban": 1, "seclusion": 5,
        },
        {
            "city": "Bali", "country": "Indonesia", "region": "Bali",
            "budget_level": "low",
            "culture": 3, "adventure": 3, "nature": 4, "beaches": 5,
            "nightlife": 5, "cuisine": 4, "wellness": 5, "urban": 2, "seclusion": 3,
        },
    ]

    print(generate_ranking_post(sample_destinations, category="nightlife", top_n=5))
