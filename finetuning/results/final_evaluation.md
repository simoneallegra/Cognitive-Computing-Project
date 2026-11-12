# Fine-tuning: final comparison (test set)

| Model | Perplexity | Grounding score | ROUGE-L |
|---|---|---|---|
| baseline_zero_shot_distilgpt2 | 30.03 | 0.000 | 0.000 |
| finetuned_context_grounded (final tool model) | 6.03 | 0.353 | 0.497 |

## Qualitative success example

### Failure case (best grounding score)
- Topic: Acapulco (review)
- Facts given: ['City: Acapulco, north_america, Mexico', "Description: Golden beaches meet vibrant nightlife, where the sound of crashing waves and lively music create an atmosphere that's both thrilling and laid-back.", 'Ideal durations: ["Short trip","Weekend","One week"]', 'Budget level: Mid-range', 'culture: 3/5', 'adventure: 4/5', 'nature: 3/5', 'beaches: 4/5', 'nightlife: 4/5', 'cuisine: 4/5', 'wellness: 3/5', 'urban: 3/5', 'seclusion: 2/5']
- Grounding score: 0.52 | ROUGE-L: 0.52
- Generated: "Acapulco, north_america — Mexico\n\nGolden beaches meet vibrant nightlife, where the sound of crashing waves and lively music create an atmosphere that's both thrilling and laid-back.\n\nArriving in Acapulco means stepping into a destination that speaks for itself. Museums, monuments and local traditions offer a solid taste of cultural identity. The local cuisine is tasty and well worth trying.\n\nFor t"
- Reference: "Acapulco, north_america — Mexico\n\nGolden beaches meet vibrant nightlife, where the sound of crashing waves and lively music create an atmosphere that's both thrilling and laid-back.\n\nArriving in Acapulco means stepping into a destination that speaks for itself. Museums, monuments and local traditions offer a solid taste of cultural identity. The food scene is rich, with traditional dishes and qual"


## Documented failure cases

### Failure case (failure case #1)
- Topic: Top 8 cities for urban (ranking)
- Facts given: ['Category: urban', '1. Milan, Italy - urban: 5/5', '2. Amsterdam, Netherlands - urban: 5/5', '3. New York, United States - urban: 5/5', '4. Chicago, United States - urban: 5/5', '5. Seoul, South Korea - urban: 5/5', '6. Washington, United States - urban: 5/5', '7. Melbourne, Australia - urban: 5/5', '8. Toronto, Canada - urban: 5/5']
- Grounding score: 0.08 | ROUGE-L: 0.42
- Generated: 'Top 8 Cities for City Life\n\nLooking for the best places to experience city life? Here are the top 8 destinations that deliver, ranked by how well they score on it.\n\n---\n\n1. **Dubai** — Dubai, United Arab Emirates\nScore: 5/5 — A true metropolis full of adventure and movement. Budget level: "Mid-range".\n\n2. **Dubai** — Dubai, United Arab Emirates\nScore: 5/5 — A true metropolis full of adventure and '
- Reference: 'Top 8 Cities for City Life\n\nLooking for the best places to experience city life? Here are the top 8 destinations that deliver, ranked by how well they score on it.\n\n---\n\n1. **Vancouver** — Vancouver, north_america, Canada\nScore: 5/5 — Full-blown metropolis energy, fast-paced and dense. Budget level: "Luxury".\n\n2. **Barcelona** — Barcelona, europe, Spain\nScore: 5/5 — Full-blown metropolis energy, f'

### Failure case (failure case #2)
- Topic: Blantyre (review)
- Facts given: ['City: Blantyre, africa, Malawi', 'Description: A city where bustling markets, friendly locals, and the aroma of street food create a vibrant and welcoming atmosphere.', 'Ideal durations: ["Short trip","Weekend"]', 'Budget level: Budget', 'culture: 3/5', 'adventure: 3/5', 'nature: 4/5', 'beaches: 1/5', 'nightlife: 2/5', 'cuisine: 3/5', 'wellness: 2/5', 'urban: 3/5', 'seclusion: 4/5']
- Grounding score: 0.24 | ROUGE-L: 0.52
- Generated: 'Blantyre, africa — Malawi\n\nA city where bustling markets, friendly locals, and the aroma of street food create a vibrant and welcoming atmosphere.\n\nArriving in Blantyre means stepping into a destination that speaks for itself. History and art permeate every corner of the city, from museums to architecture to living traditions. The food scene is rich, with traditional dishes and quality venues.\n\nFo'
- Reference: 'Blantyre, africa — Malawi\n\nA city where bustling markets, friendly locals, and the aroma of street food create a vibrant and welcoming atmosphere.\n\nArriving in Blantyre means stepping into a destination that speaks for itself. Museums, monuments and local traditions offer a solid taste of cultural identity. The local cuisine is tasty and well worth trying.\n\nFor those seeking excitement and movemen'

### Failure case (failure case #3)
- Topic: Asturias (review)
- Facts given: ['City: Asturias, europe, Spain', 'Description: Lush green landscapes meet rugged coastlines, offering a sense of tranquility and adventure with every breathtaking vista and charming village.', 'Ideal durations: ["One week","Short trip"]', 'Budget level: Mid-range', 'culture: 4/5', 'adventure: 4/5', 'nature: 5/5', 'beaches: 4/5', 'nightlife: 3/5', 'cuisine: 4/5', 'wellness: 3/5', 'urban: 2/5', 'seclusion: 4/5']
- Grounding score: 0.32 | ROUGE-L: 0.36
- Generated: 'Asturias, europe — Spain\n\nLush green landscapes meet rugged coastlines, offering a sense of tranquility and adventure with every breathtaking vista and charming village.\n\nArriving in Asturias means stepping into a destination that speaks for itself. Asturias is a concentrate of history, art and identity: every street tells a story, and the cultural heritage is simply overwhelming. The cuisine in A'
- Reference: 'Asturias, europe — Spain\n\nLush green landscapes meet rugged coastlines, offering a sense of tranquility and adventure with every breathtaking vista and charming village.\n\nArriving in Asturias means stepping into a destination that speaks for itself. History and art permeate every corner of the city, from museums to architecture to living traditions. The food scene is rich, with traditional dishes '
