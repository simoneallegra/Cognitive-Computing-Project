"""Client per il Knowledge Graph editoriale (Neo4j).

Schema:
  (:Post {id, title, post_type, created_at, summary})
  (:Topic {name})
  (:Source {url, domain})
  (:Claim {id, text})

  (Post)-[:TALKS_ABOUT]->(Topic)
  (Post)-[:USES_SOURCE]->(Source)
  (Post)-[:CONTAINS]->(Claim)
  (Claim)-[:SUPPORTED_BY]->(Source)
  (Topic)-[:RELATED_TO]->(Topic)

Il grafo viene interrogato in fase di planning/topic-suggestion (per evitare
ripetizioni e individuare gap di copertura) e in fase di drafting (per
garantire consistenza con i post precedenti e collegare il nuovo post a
topic esistenti), e viene aggiornato in modo incrementale solo dopo
l'approvazione umana di un post.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from neo4j import GraphDatabase

import config


class KnowledgeGraph:
    def __init__(self, uri: str | None = None, user: str | None = None, password: str | None = None):
        self.driver = GraphDatabase.driver(
            uri or config.NEO4J_URI,
            auth=(user or config.NEO4J_USER, password or config.NEO4J_PASSWORD),
        )
        self._ensure_constraints()

    def close(self) -> None:
        self.driver.close()

    def __enter__(self) -> "KnowledgeGraph":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    # ------------------------------------------------------------------ setup
    def _ensure_constraints(self) -> None:
        constraints = [
            "CREATE CONSTRAINT post_id IF NOT EXISTS FOR (p:Post) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT topic_name IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE",
            "CREATE CONSTRAINT source_url IF NOT EXISTS FOR (s:Source) REQUIRE s.url IS UNIQUE",
            "CREATE CONSTRAINT claim_id IF NOT EXISTS FOR (c:Claim) REQUIRE c.id IS UNIQUE",
        ]
        with self.driver.session() as session:
            for cypher in constraints:
                session.run(cypher)

    # ------------------------------------------------------------------ writes (solo dopo approvazione)
    def add_post(self, post_id: str, title: str, post_type: str, summary: str = "") -> None:
        query = """
        MERGE (p:Post {id: $id})
        SET p.title = $title,
            p.post_type = $post_type,
            p.summary = $summary,
            p.created_at = $created_at
        """
        with self.driver.session() as session:
            session.run(
                query,
                id=post_id,
                title=title,
                post_type=post_type,
                summary=summary,
                created_at=datetime.now(timezone.utc).isoformat(),
            )

    def link_post_topic(self, post_id: str, topic: str) -> None:
        query = """
        MATCH (p:Post {id: $post_id})
        MERGE (t:Topic {name: $topic})
        MERGE (p)-[:TALKS_ABOUT]->(t)
        """
        with self.driver.session() as session:
            session.run(query, post_id=post_id, topic=topic)

    def link_topics_related(self, topic_a: str, topic_b: str) -> None:
        if topic_a == topic_b:
            return
        query = """
        MERGE (a:Topic {name: $a})
        MERGE (b:Topic {name: $b})
        MERGE (a)-[:RELATED_TO]-(b)
        """
        with self.driver.session() as session:
            session.run(query, a=topic_a, b=topic_b)

    def link_post_source(self, post_id: str, url: str, domain: str = "") -> None:
        query = """
        MATCH (p:Post {id: $post_id})
        MERGE (s:Source {url: $url})
        SET s.domain = coalesce($domain, s.domain)
        MERGE (p)-[:USES_SOURCE]->(s)
        """
        with self.driver.session() as session:
            session.run(query, post_id=post_id, url=url, domain=domain)

    def add_claim(self, post_id: str, claim_id: str, text: str, source_url: str | None = None) -> None:
        query = """
        MATCH (p:Post {id: $post_id})
        MERGE (c:Claim {id: $claim_id})
        SET c.text = $text
        MERGE (p)-[:CONTAINS]->(c)
        """
        with self.driver.session() as session:
            session.run(query, post_id=post_id, claim_id=claim_id, text=text)
            if source_url:
                session.run(
                    """
                    MATCH (c:Claim {id: $claim_id})
                    MERGE (s:Source {url: $source_url})
                    MERGE (c)-[:SUPPORTED_BY]->(s)
                    """,
                    claim_id=claim_id,
                    source_url=source_url,
                )

    # ------------------------------------------------------------------ queries (planning / drafting)
    def get_recent_topics(self, limit: int = 10) -> list[str]:
        """Topic trattati piu' di recente: usati per EVITARE ripetizione."""
        query = """
        MATCH (p:Post)-[:TALKS_ABOUT]->(t:Topic)
        RETURN t.name AS topic, max(p.created_at) AS last_used
        ORDER BY last_used DESC
        LIMIT $limit
        """
        with self.driver.session() as session:
            result = session.run(query, limit=limit)
            return [r["topic"] for r in result]

    def get_coverage_gaps(self, candidate_topics: list[str]) -> list[str]:
        """Topic candidati dal seed del dominio che non sono MAI stati trattati."""
        query = """
        UNWIND $candidates AS candidate
        OPTIONAL MATCH (t:Topic {name: candidate})<-[:TALKS_ABOUT]-(:Post)
        WITH candidate, count(t) AS uses
        WHERE uses = 0
        RETURN candidate AS topic
        """
        with self.driver.session() as session:
            result = session.run(query, candidates=candidate_topics)
            return [r["topic"] for r in result]

    def get_related_context(self, topic: str, limit: int = 5) -> list[dict[str, Any]]:
        """Post precedenti collegati allo stesso topic o a topic correlati: usati
        in fase di drafting per garantire consistenza e collegare il nuovo post."""
        query = """
        MATCH (t:Topic {name: $topic})
        OPTIONAL MATCH (t)-[:RELATED_TO]-(related:Topic)
        WITH collect(DISTINCT t.name) + collect(DISTINCT related.name) AS topics
        UNWIND topics AS topic_name
        MATCH (p:Post)-[:TALKS_ABOUT]->(:Topic {name: topic_name})
        RETURN DISTINCT p.id AS id, p.title AS title, p.summary AS summary, p.post_type AS post_type
        ORDER BY p.created_at DESC
        LIMIT $limit
        """
        with self.driver.session() as session:
            result = session.run(query, topic=topic, limit=limit)
            return [dict(r) for r in result]

    def expand_query_topics(self, topic: str, limit: int = 5) -> list[str]:
        """Topic correlati nel KG: usati per espandere/raffinare la query di
        retrieval nel pattern K-RAG (knowledge-graph-augmented RAG)."""
        query = """
        MATCH (t:Topic {name: $topic})-[:RELATED_TO]-(related:Topic)
        RETURN DISTINCT related.name AS topic
        LIMIT $limit
        """
        with self.driver.session() as session:
            result = session.run(query, topic=topic, limit=limit)
            return [r["topic"] for r in result]

    def get_claims_for_topic(self, topic: str, limit: int = 10) -> list[str]:
        query = """
        MATCH (:Topic {name: $topic})<-[:TALKS_ABOUT]-(p:Post)-[:CONTAINS]->(c:Claim)
        RETURN DISTINCT c.text AS text
        LIMIT $limit
        """
        with self.driver.session() as session:
            result = session.run(query, topic=topic, limit=limit)
            return [r["text"] for r in result]

    def summary(self) -> dict[str, Any]:
        """Riassunto sintetico del KG, usato per popolare lo stato dell'agente."""
        query = """
        MATCH (p:Post) WITH count(p) AS posts
        MATCH (t:Topic) WITH posts, count(t) AS topics
        MATCH (s:Source) WITH posts, topics, count(s) AS sources
        MATCH (c:Claim) RETURN posts, topics, sources, count(c) AS claims
        """
        with self.driver.session() as session:
            result = session.run(query).single()
            if result is None:
                return {"posts": 0, "topics": 0, "sources": 0, "claims": 0}
            return dict(result)


if __name__ == "__main__":
    # Debug: `python kg/client.py` verifica la connessione a Neo4j e mostra
    # lo stato attuale del grafo, senza dover passare da tutto l'agente.
    print("Test di connessione a Neo4j...")
    try:
        with KnowledgeGraph() as kg:
            print("Connessione OK.\n")
            print("summary():", kg.summary())
            print("recent_topics(limit=5):", kg.get_recent_topics(limit=5))
            print(
                "coverage_gaps (esempio):",
                kg.get_coverage_gaps(["culture", "adventure", "beaches"]),
            )
    except Exception as exc:
        print(f"Connessione fallita: {exc}")
        print("Controlla che Neo4j sia avviato (./runtime/start_neo4j.sh) "
              "e che NEO4J_URI/USER/PASSWORD in .env siano corretti.")
