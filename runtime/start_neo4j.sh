#!/usr/bin/env bash
# Avvia Neo4j Community Edition usando il JDK 17 portatile scaricato in
# questa stessa cartella (nessun permesso di root necessario).
set -euo pipefail

RUNTIME_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export JAVA_HOME="$RUNTIME_DIR/jdk17"
export PATH="$JAVA_HOME/bin:$PATH"

NEO4J_BIN="$RUNTIME_DIR/neo4j-community-5.26.0/bin/neo4j"

if (echo > /dev/tcp/127.0.0.1/7687) 2>/dev/null; then
    echo "Neo4j is already running on bolt://localhost:7687"
    exit 0
fi

echo "Starting Neo4j (JAVA_HOME=$JAVA_HOME)..."
nohup "$NEO4J_BIN" console > "$RUNTIME_DIR/neo4j.log" 2>&1 &
disown
echo "Neo4j starting in background (PID $!). Logs: $RUNTIME_DIR/neo4j.log"
echo "Waiting for bolt port to come up..."
for _ in $(seq 1 30); do
    if (echo > /dev/tcp/127.0.0.1/7687) 2>/dev/null; then
        echo "Neo4j is up on bolt://localhost:7687"
        exit 0
    fi
    sleep 1
done
echo "Neo4j did not come up within 30s, check $RUNTIME_DIR/neo4j.log"
exit 1
