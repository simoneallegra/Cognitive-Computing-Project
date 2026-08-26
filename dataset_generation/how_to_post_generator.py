from __future__ import annotations
import ast
import json

def _normalize_durations(durations):
    """Normalizza `durations` in una lista di stringhe, qualunque sia il
    formato in ingresso (lista vera, stringa JSON, stringa singola)"""
    if durations is None:
        return []

    if isinstance(durations, (list, tuple)):
        normalized = []
        for item in durations:
            normalized.append(str(item).strip())
        return normalized

    if isinstance(durations, str):
        text = durations.strip()
        if text.startswith("[") and text.endswith("]"):
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(text)
                    if isinstance(parsed, (list, tuple)):
                        normalized = []
                        for item in parsed:
                            normalized.append(str(item).strip())
                        return normalized
                except (ValueError, SyntaxError):
                    continue
        return [text] if text else []

    return [str(durations)]


def _clamp_score(value) -> int:
    """Forza il punteggio nel range 1-5"""
    try:
        value = int(value)
    except (TypeError, ValueError):
        return 3
    return max(1, min(5, value))


_DURATION_DAYS_MAP = [
    ("weekend", 2),
    ("short trip", 3),
    ("one week", 7),
    ("week", 7),
    ("long stay", 10),
]


def _days_for_duration(durations) -> int:
    """Sceglie i giorni di itinerario dalla keyword di durata piu' lunga
    trovata in `durations`, default 3"""
    best_match = 0
    for raw in durations:
        lowered = raw.lower()
        for keyword, days in _DURATION_DAYS_MAP:
            if keyword in lowered:
                best_match = max(best_match, days)
    return best_match if best_match else 3


def _duration_headline(durations) -> str:
    """Costruisce l'etichetta "X giorni" per l'headline"""
    days = _days_for_duration(durations)
    if days == 7:
        return "1 Week"
    if days == 14:
        return "2 Weeks"
    return f"{days} Days"



CULTURE_BLOCKS = {
    1: "Culture isn't the main draw here, so don't stress about museum-hopping — a quick stop at the main square is enough.",
    2: "Set aside an hour for the most central historic site; that's plenty to get a feel for local history.",
    3: "Block out half a day for the main museum or old town walk — it's worth the time.",
    4: "Dedicate a full morning to the historic center and at least one major museum; {city} has real depth here.",
    5: "Give {city}'s culture its own full day: start at the top museum, wander the old town, and don't rush — there's history at every turn.",
}

ADVENTURE_BLOCKS = {
    1: "Skip the adventure activities — they're not really {city}'s thing.",
    2: "If you've got a spare afternoon, one easy outdoor activity is worth squeezing in.",
    3: "Book a half-day hike or outdoor excursion — a solid way to break up the city time.",
    4: "Plan a full day for an adventure activity (hiking, watersports, climbing) — {city} rewards it.",
    5: "Build an entire day, maybe two, around adventure: {city} is genuinely one of the best spots for it, so don't half-do this.",
}

NATURE_BLOCKS = {
    1: "Don't expect much from the natural scenery — focus your time elsewhere.",
    2: "A short stroll through a local park is enough nature for this trip.",
    3: "Set aside a couple of hours for a scenic spot just outside the center.",
    4: "Reserve a half-day for the natural landscapes nearby — bring a camera.",
    5: "Dedicate a full day to the surrounding nature; the landscapes around {city} are a highlight you shouldn't compress into a quick stop.",
}

BEACHES_BLOCKS = {
    1: "leave the swimsuit at home — beaches aren't part of this itinerary",
    2: "if time allows, a quick stop at the nearest beach is a nice add-on, not a priority",
    3: "carve out an afternoon for the coastline — good for a relaxed break",
    4: "set aside at least half a day for the beach — it's one of the trip's highlights",
    5: "block out a full beach day (or two) — the coastline here is too good to rush",
}

NIGHTLIFE_BLOCKS = {
    1: "Don't plan around nightlife — most things wind down early here.",
    2: "One casual evening at a local bar is enough; don't expect much beyond that.",
    3: "Pick one night to explore the bar scene — a fun, low-key evening out.",
    4: "Keep at least two evenings free for bars and venues — the nightlife here deserves real time.",
    5: "Reserve your evenings, plural: {city}'s nightlife is a core part of the experience, not an afterthought.",
}

CUISINE_BLOCKS = {
    1: "Keep meals simple and quick — food isn't the highlight of this trip.",
    2: "Try one well-reviewed local spot, but don't build the itinerary around meals.",
    3: "Book at least one proper sit-down meal at a well-regarded local restaurant.",
    4: "Plan two or three meals around well-known local restaurants — the food scene earns it.",
    5: "Build the itinerary around food: book ahead at top local restaurants, because {city}'s cuisine is reason enough to visit.",
}

WELLNESS_BLOCKS = {
    1: "Don't expect a wellness-focused trip — there's little on offer here.",
    2: "A short spa visit is possible, but it's a minor add-on, not a focus.",
    3: "Set aside a few hours at a spa or wellness center for a relaxed mid-trip break.",
    4: "Reserve at least half a day for a proper spa or wellness experience.",
    5: "Build in a full rest day at a spa or wellness retreat — {city} is genuinely set up for it, so don't skip it.",
}

URBAN_BLOCKS = {
    1: "This is a rural, low-key destination — don't expect city infrastructure or fast-paced energy.",
    2: "Keep expectations modest on the city-life front; it's a quiet, low-density place.",
    3: "There's a decent mix of city buzz and quiet streets — easy to navigate either way.",
    4: "Expect a busy, metropolitan pace — factor in extra time for traffic and crowds.",
    5: "Prepare for full metropolis energy: {city} moves fast, so plan transport and timing carefully.",
}

SECLUSION_BLOCKS = {
    1: "Expect crowds pretty much everywhere — this isn't a place to escape the tourist trail.",
    2: "Quiet corners exist but take some searching; don't count on finding them by accident.",
    3: "It's fairly easy to find a quiet spot away from the main crowds if you want a break.",
    4: "Build in some downtime away from the main sights — {city} has real secluded spots worth seeking out.",
    5: "If true peace and quiet matter to you, this is one of the best picks — seek out the secluded spots deliberately, they're worth it.",
}

CATEGORY_LABELS = {
    "culture": "culture",
    "adventure": "adventure",
    "nature": "nature",
    "beaches": "the beaches",
    "nightlife": "the nightlife",
    "cuisine": "the food",
    "wellness": "wellness",
    "urban": "city life",
    "seclusion": "quiet, secluded spots",
}

ACTIVITY_BLOCKS = {
    "culture": CULTURE_BLOCKS,
    "adventure": ADVENTURE_BLOCKS,
    "nature": NATURE_BLOCKS,
    "beaches": BEACHES_BLOCKS,
    "nightlife": NIGHTLIFE_BLOCKS,
    "cuisine": CUISINE_BLOCKS,
    "wellness": WELLNESS_BLOCKS,
    "urban": URBAN_BLOCKS,
    "seclusion": SECLUSION_BLOCKS,
}


def _block_for(category: str, score: int, city: str) -> str:
    """Prende il consiglio per categoria+punteggio, sostituendo {city}"""
    score = _clamp_score(score)
    text = ACTIVITY_BLOCKS[category][score].format(city=city)
    return text[0].upper() + text[1:] if text else text


def _day_title_label(category: str) -> str:
    """Etichetta pulita per il titolo del giorno: toglie l'articolo
    iniziale ("the nightlife" -> "Nightlife")"""
    label = CATEGORY_LABELS[category]
    for article in ("the ", "The "):
        if label.startswith(article):
            label = label[len(article):]
    return label[0].upper() + label[1:]


def _ranked_categories(data):
    """Categorie ordinate dal punteggio piu' alto al piu' basso"""
    scores = {}
    for cat in ACTIVITY_BLOCKS:
        scores[cat] = _clamp_score(data.get(cat, 3))
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def _budget_line(budget_level: str) -> str:
    """Riporta budget_level cosi' come arriva, senza tradurlo"""
    return str(budget_level) if budget_level else ""


def _duration_list_text(durations) -> str:
    """Riporta ideal_durations cosi' come arriva, unendo i valori"""
    if not durations:
        return ""
    if len(durations) == 1:
        return durations[0]
    if len(durations) == 2:
        return f"{durations[0]} and {durations[1]}"
    return ", ".join(durations[:-1]) + f" and {durations[-1]}"


def _build_day_plan(ranked, num_days: int, city: str):
    """Distribuisce le categorie top-ranked sui giorni: quelle con
    punteggio piu' alto hanno un giorno dedicato, le altre si raggruppano"""

    def _score_for(cat):
        for c, s in ranked:
            if c == cat:
                return s
        return None

    # tiene solo le categorie con punteggio almeno 2
    relevant = []
    for cat, score in ranked:
        if score >= 2:
            relevant.append(cat)
    if not relevant:
        relevant = [ranked[0][0]]  # fallback: mostra sempre almeno una categoria

    days = []
    cat_index = 0

    for day_num in range(1, num_days + 1):
        if cat_index >= len(relevant):
            # categorie finite: richiama la migliore con un tono piu' leggero
            top_cat = ranked[0][0]
            score = _score_for(top_cat)
            line = _block_for(top_cat, score, city)
            days.append(
                f"**Day {day_num}: Take it slow**\n"
                f"No must-do left unchecked — use this day as a buffer. Revisit a favorite spot, "
                f"or simply slow down: {line}"
            )
            continue

        # ultimo giorno: raggruppa le categorie rimaste
        is_last_day = day_num == num_days
        remaining = len(relevant) - cat_index

        if is_last_day and remaining > 1:
            chunk = relevant[cat_index:]
            cat_index = len(relevant)
            lines = []
            for cat in chunk:
                score = _score_for(cat)
                lines.append(f"- {_block_for(cat, score, city)}")
            wrap_title_parts = []
            for c in chunk:
                wrap_title_parts.append(_day_title_label(c))
            wrap_title = ", ".join(wrap_title_parts)
            days.append(
                f"**Day {day_num}: Wrap-up — {wrap_title}**\n"
                + "\n".join(lines)
            )
        else:
            cat = relevant[cat_index]
            cat_index += 1
            score = _score_for(cat)
            label = _day_title_label(cat)
            days.append(
                f"**Day {day_num}: {label}**\n{_block_for(cat, score, city)}"
            )

    return days


def generate_howto(data) -> str:
    """Genera il post "How to spend X days in {city}", stile itinerario
    giorno per giorno guidato dai punteggi 1-5 delle categorie"""
    city = data.get("city", "this destination")
    region = data.get("region", "")
    country = data.get("country", "")
    short_description = data.get("short_description", "")
    budget_level = _budget_line(data.get("budget_level", ""))

    durations = _normalize_durations(data.get("ideal_durations"))
    num_days = _days_for_duration(durations)
    headline_duration = _duration_headline(durations)
    duration_list_text = _duration_list_text(durations)

    ranked = _ranked_categories(data)
    top_cat, _ = ranked[0]
    top_label = CATEGORY_LABELS[top_cat]

    day_plan = _build_day_plan(ranked, num_days, city)
    day_plan_text = "\n\n".join(day_plan)

    # riga location: gestisce regione mancante o duplicata
    if region and region.lower() != city.lower():
        location_line = f"{city}, {region}, {country}"
    else:
        location_line = f"{city}, {country}"

    intro = (
        f"{short_description} " if short_description else ""
    ) + (
        f"If you're planning a trip built around {duration_list_text}, here's exactly how to spend "
        f"your time — with {top_label} leading the way."
    )

    post = f"""How to Spend {headline_duration} in {city}

Location: {location_line}
Budget level: "{budget_level}"
Best for: "{duration_list_text}"

{intro}

---

{day_plan_text}

---

**Bottom line:** {city} is best enjoyed by leaning into {top_label} — build your schedule around it first, then fill the gaps with whatever else fits your pace and budget."""

    return post.strip()
