#!/usr/bin/env python3
"""Initialize Neo4j schema (constraints + indexes).

Run after starting Neo4j for the first time:
    python scripts/setup_neo4j.py
"""

from src.graph.client import GraphClient
from src.graph.schema import init_schema, print_schema_summary


def main():
    with GraphClient() as client:
        print(f"Connected to {client.verify()}")
        init_schema(client)
        print_schema_summary(client)
        print("\nNeo4j schema ready.")


if __name__ == "__main__":
    main()
