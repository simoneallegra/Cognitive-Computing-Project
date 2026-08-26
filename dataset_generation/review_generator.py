from __future__ import annotations
import json


CULTURE_SENTENCES = {
    1: "The local culture stays in the background, barely visible to the passing visitor.",
    2: "There are traces of history and tradition, but they take a bit of patience to find.",
    3: "Museums, monuments and local traditions offer a solid taste of cultural identity.",
    4: "History and art permeate every corner of the city, from museums to architecture to living traditions.",
    5: "{city} is a concentrate of history, art and identity: every street tells a story, "
       "and the cultural heritage is simply overwhelming.",
}

ADVENTURE_SENTENCES = {
    1: "Thrill-seekers will struggle to find much here.",
    2: "A few outdoor activities are available, but the offer remains limited.",
    3: "Hiking and outdoor activities are not in short supply for those who want to get moving.",
    4: "Adventure lovers will find plenty to do, between trails and dynamic activities.",
    5: "For adrenaline junkies, {city} is a true open-air playground, "
       "packed with activities and intense experiences.",
}

NATURE_SENTENCES = {
    1: "The natural landscape is not the area's strong point.",
    2: "There are green spaces, but nothing particularly memorable.",
    3: "The surrounding nature offers pleasant views and the occasional surprise.",
    4: "The natural landscapes are one of the best reasons to visit this area.",
    5: "Nature is the absolute protagonist here, with views that take your breath away.",
}

BEACHES_SENTENCES = {
    1: "anyone dreaming of beaches and sea will be disappointed",
    2: "there are beaches, but they're not the reason to come here",
    3: "the coastline offers good opportunities for relaxing and swimming",
    4: "the beaches are among the most beautiful you could wish for in the area",
    5: "the beaches are simply postcard-perfect, among the most spectacular in the region",
}

NIGHTLIFE_SENTENCES = {
    1: "After dinner, the city empties out almost completely.",
    2: "There are a few venues open, but the atmosphere stays quiet.",
    3: "Bars and venues bring the evenings to life in a discreet but pleasant way.",
    4: "Nightlife is lively, with venues and music keeping the streets buzzing late into the night.",
    5: "Nights in {city} are simply legendary, full of venues, music and non-stop energy.",
}

CUISINE_SENTENCES = {
    1: "The local cuisine won't leave much of a memory.",
    2: "There are a few interesting dishes, but the food scene remains average overall.",
    3: "The local cuisine is tasty and well worth trying.",
    4: "The food scene is rich, with traditional dishes and quality venues.",
    5: "The cuisine in {city} deserves a chapter of its own: a memorable culinary experience, "
       "reason enough on its own to justify the trip.",
}

WELLNESS_SENTENCES = {
    1: "Opportunities to relax are practically nonexistent.",
    2: "A few spas or wellness centers exist, but only marginally.",
    3: "There's no shortage of chances to enjoy some relaxation and wellness.",
    4: "Spas, wellness centers and a slower pace make the stay truly rejuvenating.",
    5: "{city} is a true wellness sanctuary, perfect for anyone who wants to "
       "genuinely unwind body and mind.",
}

URBAN_SENTENCES = {
    1: "The atmosphere is distinctly rural, far from any city bustle.",
    2: "The urban fabric is minimal, almost peripheral.",
    3: "The city offers a good balance between urban life and quieter spaces.",
    4: "Metropolitan energy is strongly felt, between skyscrapers, traffic and city rhythms.",
    5: "{city} is a metropolis in every sense of the word: dynamic, dense and constantly on the move.",
}

SECLUSION_SENTENCES = {
    1: "There's certainly no seclusion here: mass tourism is the norm.",
    2: "Quieter spots do exist, but they're hard to find.",
    3: "It's possible to carve out pockets of peace away from the crowds.",
    4: "Those seeking tranquility will find several secluded, uncrowded spots.",
    5: "For anyone wanting true seclusion, far from everything and everyone, this is the place.",
}

CATEGORY_LABELS = {
    "culture": "culture",
    "adventure": "adventure",
    "nature": "nature",
    "beaches": "the beaches",
    "nightlife": "the nightlife",
    "cuisine": "the cuisine",
    "wellness": "wellness",
    "urban": "city life",
    "seclusion": "tranquility",
}

SENTENCE_BANKS = {
    "culture": CULTURE_SENTENCES,
    "adventure": ADVENTURE_SENTENCES,
    "nature": NATURE_SENTENCES,
    "beaches": BEACHES_SENTENCES,
    "nightlife": NIGHTLIFE_SENTENCES,
    "cuisine": CUISINE_SENTENCES,
    "wellness": WELLNESS_SENTENCES,
    "urban": URBAN_SENTENCES,
    "seclusion": SECLUSION_SENTENCES,
}


def _sentence_for(category: str, score: int, city: str, lowercase_start: bool = False) -> str:
    """Prende la frase dal banco per categoria+punteggio, sostituendo
    {city} — se lowercase_start=True la prima lettera va minuscola (a
    meno che non inizi col nome della citta')"""
    bank = SENTENCE_BANKS[category]
    sentence = bank[score].format(city=city)

    if lowercase_start and sentence and not sentence.startswith(city):
        sentence = sentence[0].lower() + sentence[1:]

    return sentence


def _ideal_durations_sentence(durations: str) -> str:
    """Riporta ideal_durations cosi' come arriva, senza parafrasare — piu'
    valori si uniscono con virgole (e "and" prima dell'ultimo)"""
    if not durations:
        return ""
    durations = json.loads(durations)

    if len(durations) == 1:
        return durations[0]
    if len(durations) == 2:
        return f"{durations[0]} or {durations[1]}"
    return ", ".join(durations[:-1]) + f" or {durations[-1]}"


def _budget_sentence(budget_level: str) -> str:
    """Riporta budget_level cosi' come arriva, senza tradurlo"""
    return str(budget_level) if budget_level else ""


def _top_categories_summary(data, city: str) -> str:
    """Trova le 2 categorie con punteggio piu' alto e costruisce la chiusura"""
    scores = {}
    for cat in SENTENCE_BANKS:
        scores[cat] = data.get(cat, 3)

    top_two = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:2]

    labels = []
    for cat, _ in top_two:
        labels.append(CATEGORY_LABELS[cat])

    if len(labels) == 2:
        categories_text = f"{labels[0]} and {labels[1]}"
    else:
        categories_text = labels[0] if labels else "its many qualities"

    return (
        f"{city} stands out above all for {categories_text}, "
        f"making it a destination well worth considering for your next trip."
    )


def generate_review(data) -> str:
    """Genera il testo completo della recensione dal dizionario `data`"""
    city = data.get("city", "this destination")
    region = data.get("region", "")
    country = data.get("country", "")
    short_description = data.get("short_description", "")
    ideal_durations = data.get("ideal_durations", [])
    budget_level = data.get("budget_level", "medium")

    culture_sentence = _sentence_for("culture", data.get("culture", 3), city)
    adventure_sentence = _sentence_for("adventure", data.get("adventure", 3), city)
    nature_sentence = _sentence_for("nature", data.get("nature", 3), city)
    beaches_sentence = _sentence_for("beaches", data.get("beaches", 3), city)
    nightlife_sentence = _sentence_for("nightlife", data.get("nightlife", 3), city)
    cuisine_sentence = _sentence_for("cuisine", data.get("cuisine", 3), city)
    wellness_sentence = _sentence_for("wellness", data.get("wellness", 3), city, lowercase_start=True)
    urban_sentence = _sentence_for("urban", data.get("urban", 3), city, lowercase_start=True)
    seclusion_sentence = _sentence_for("seclusion", data.get("seclusion", 3), city)

    duration_sentence = _ideal_durations_sentence(ideal_durations)
    budget_sentence = _budget_sentence(budget_level)
    summary_sentence = _top_categories_summary(data, city)

    # heading: gestisce regione mancante o uguale alla citta'
    if region and region.lower() != city.lower():
        heading = f"{city}, {region} — {country}"
    else:
        heading = f"{city} — {country}"

    review = f"""{heading}

{short_description}

Arriving in {city} means stepping into a destination that speaks for itself. {culture_sentence} {cuisine_sentence}

For those seeking excitement and movement, {city} delivers: {adventure_sentence} {nature_sentence} Rounding things out, {beaches_sentence}.

When the sun goes down, the city changes face. {nightlife_sentence} For those who prefer a slower pace instead, {wellness_sentence}

From an architectural and social standpoint, {urban_sentence} {seclusion_sentence}

In practical terms, {city} works well for a {duration_sentence} type of trip, with a "{budget_sentence}" budget level.

In short: {summary_sentence}"""

    return review.strip()
