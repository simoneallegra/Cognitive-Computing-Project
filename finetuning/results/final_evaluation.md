# Fine-tuning: final comparison (test set)

| Model | Perplexity | Grounding score | ROUGE-L |
|---|---|---|---|
| baseline_zero_shot_distilgpt2 | 30.91 | 0.000 | 0.000 |
| legacy_finetuned_no_context | 1.84 | 0.185 | 0.170 |
| finetuned_context_grounded (final tool model) | 1.27 | 0.349 | 0.495 |

## Qualitative success example

### Failure case (best grounding score)
- Topic: Nicosia (review)
- Facts given: ['City: Nicosia, europe, Cyprus', 'Description: Wander through leafy streets where ancient history meets modern life, with a vibrant café culture and a warm, welcoming atmosphere.', 'Ideal durations: ["One week","Short trip","Weekend"]', 'Budget level: Mid-range', 'culture: 4/5', 'adventure: 3/5', 'nature: 3/5', 'beaches: 2/5', 'nightlife: 3/5', 'cuisine: 4/5', 'wellness: 3/5', 'urban: 4/5', 'seclusion: 2/5']
- Grounding score: 0.46 | ROUGE-L: 0.48
- Generated: 'Nicosia, europe — Cyprus\n\nWander through leafy streets where ancient history meets modern life, with a vibrant café culture and a warm, welcoming atmosphere.\n\nArriving in Nicosia means stepping into a destination that speaks for itself. History and art permeate every corner of the city, from museums to architecture to living traditions. The food scene is rich, with traditional dishes and quality v'
- Reference: 'Nicosia, europe — Cyprus\n\nWander through leafy streets where ancient history meets modern life, with a vibrant café culture and a warm, welcoming atmosphere.\n\nArriving in Nicosia means stepping into a destination that speaks for itself. History and art permeate every corner of the city, from museums to architecture to living traditions. The food scene is rich, with traditional dishes and quality v'


## Documented failure cases

### Failure case (failure case #1)
- Topic: San Pedro de Atacama (review)
- Facts given: ['City: San Pedro de Atacama, south_america, Chile', 'Description: Endless desert vistas meet surreal landscapes, where clear night skies reveal a tapestry of stars, creating an otherworldly sense of tranquility and wonder.', 'Ideal durations: ["One week","Short trip"]', 'Budget level: Mid-range', 'culture: 3/5', 'adventure: 5/5', 'nature: 5/5', 'beaches: 1/5', 'nightlife: 2/5', 'cuisine: 3/5', 'wellness: 3/5', 'urban: 2/5', 'seclusion: 5/5']
- Grounding score: 0.30 | ROUGE-L: 0.44
- Generated: 'San Pedro de Atacama, south_america — Chile\n\nEndless desert vistas meet surreal landscapes, where clear night skies reveal a tapestry of stars, creating an otherworldly sense of tranquility and wonder.\n\nArriving in San Pedro de Atacama means stepping into a destination that speaks for itself. History and art permeate every corner of the city, from museums to architecture to living traditions. The '
- Reference: 'San Pedro de Atacama, south_america — Chile\n\nEndless desert vistas meet surreal landscapes, where clear night skies reveal a tapestry of stars, creating an otherworldly sense of tranquility and wonder.\n\nArriving in San Pedro de Atacama means stepping into a destination that speaks for itself. Museums, monuments and local traditions offer a solid taste of cultural identity. The local cuisine is tas'

### Failure case (failure case #2)
- Topic: Koh Lanta (review)
- Facts given: ['City: Koh Lanta, asia, Thailand', 'Description: Golden beaches, gentle waves, and lush greenery create a serene escape where you can unwind and soak in the tranquil island vibe.', 'Ideal durations: ["Weekend","One week","Short trip"]', 'Budget level: Budget', 'culture: 3/5', 'adventure: 4/5', 'nature: 4/5', 'beaches: 5/5', 'nightlife: 3/5', 'cuisine: 4/5', 'wellness: 4/5', 'urban: 2/5', 'seclusion: 4/5']
- Grounding score: 0.31 | ROUGE-L: 0.44
- Generated: 'Koh Lanta, asia — Thailand\n\nGolden beaches, gentle waves, and lush greenery create a serene escape where you can unwind and soak in the tranquil island vibe.\n\nArriving in Koh Lanta means stepping into a destination that speaks for itself. Koh Lanta is a concentrate of history, art and identity: every street tells a story, and the cultural heritage is simply overwhelming. The food scene is rich, wi'
- Reference: 'Koh Lanta, asia — Thailand\n\nGolden beaches, gentle waves, and lush greenery create a serene escape where you can unwind and soak in the tranquil island vibe.\n\nArriving in Koh Lanta means stepping into a destination that speaks for itself. Museums, monuments and local traditions offer a solid taste of cultural identity. The food scene is rich, with traditional dishes and quality venues.\n\nFor those '

### Failure case (failure case #3)
- Topic: Siem Reap (review)
- Facts given: ['City: Siem Reap, asia, Cambodia', 'Description: Ancient temples rise through the jungle mist, while bustling markets and vibrant street food create a lively, sensory-rich experience.', 'Ideal durations: ["Short trip","Weekend","One week"]', 'Budget level: Budget', 'culture: 5/5', 'adventure: 3/5', 'nature: 4/5', 'beaches: 1/5', 'nightlife: 3/5', 'cuisine: 4/5', 'wellness: 3/5', 'urban: 3/5', 'seclusion: 3/5']
- Grounding score: 0.38 | ROUGE-L: 0.41
- Generated: 'Siem Reap, asia — Cambodia\n\nAncient temples rise through the jungle mist, while bustling markets and vibrant street food create a lively, sensory-rich experience.\n\nArriving in Siem Reap means stepping into a destination that speaks for itself. History and art permeate every corner of the city, from museums to architecture to living traditions. The food scene is rich, with traditional dishes and qu'
- Reference: 'Siem Reap, asia — Cambodia\n\nAncient temples rise through the jungle mist, while bustling markets and vibrant street food create a lively, sensory-rich experience.\n\nArriving in Siem Reap means stepping into a destination that speaks for itself. Siem Reap is a concentrate of history, art and identity: every street tells a story, and the cultural heritage is simply overwhelming. The food scene is ric'
