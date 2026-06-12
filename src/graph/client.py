"""Neo4j client wrapper.

Provides a thin interface over the official neo4j Python driver so that
every agent uses the same connection logic, transaction handling, and
MERGE patterns.

Usage:
    from src.graph.client import GraphClient

    with GraphClient() as client:
        client.merge_node("Source", {"name": "Artemisia annua"}, extra={...})
        client.merge_edge("Source", {"name": "Artemisia annua"},
                          "Traditional_Malady", {"name": "Intermittent fever"},
                          "TREATS_TRADITIONALLY", {"confidence_score": 0.9})
        results = client.run("MATCH (s:Source) RETURN s.name")
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from neo4j import GraphDatabase, Driver, Session

load_dotenv()


class GraphClient:
    """Shared Neo4j client used by all agents."""

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
    ):
        self._uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self._user = user or os.getenv("NEO4J_USER", "neo4j")
        self._password = password or os.getenv("NEO4J_PASSWORD", "password")
        self._database = database or os.getenv("NEO4J_DATABASE", "neo4j")
        self._driver: Driver | None = None

    # -- lifecycle -------------------------------------------------------------

    def connect(self) -> GraphClient:
        """Create the driver connection."""
        self._driver = GraphDatabase.driver(
            self._uri, auth=(self._user, self._password)
        )
        return self

    def close(self) -> None:
        if self._driver:
            self._driver.close()
            self._driver = None

    def __enter__(self) -> GraphClient:
        return self.connect()

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @property
    def driver(self) -> Driver:
        if self._driver is None:
            raise RuntimeError("Not connected — call .connect() or use `with`")
        return self._driver

    def verify(self) -> dict:
        """Verify the connection and return server info."""
        self.driver.verify_connectivity()
        info = self.driver.get_server_info()
        return {"address": str(info.address), "agent": info.agent}

    # -- query execution -------------------------------------------------------

    def run(self, query: str, params: dict | None = None) -> list[dict]:
        """Execute a Cypher query and return results as list of dicts."""
        with self.driver.session(database=self._database) as session:
            result = session.run(query, params or {})
            return [record.data() for record in result]

    def run_write(self, query: str, params: dict | None = None) -> list[dict]:
        """Execute a write query inside an explicit write transaction."""
        with self.driver.session(database=self._database) as session:
            return session.execute_write(
                lambda tx: [r.data() for r in tx.run(query, params or {})]
            )

    def batched_unwind_write(
        self,
        query: str,
        rows: list[dict],
        *,
        batch_size: int = 500,
        param_name: str = "rows",
    ) -> int:
        """Apply a parameterized UNWIND query in client-side batches.

        `query` should reference `$rows` (or whatever `param_name` is) as
        the UNWIND list. Each batch is one Neo4j transaction, replacing
        N per-row round trips with one round trip per batch — the
        standard pattern for write-heavy workloads on hosted Neo4j
        (AuraDB), where per-tx network RTT dominates wall clock.

        Returns the total number of rows written. Empty `rows` is a
        no-op.
        """
        if not rows:
            return 0
        total = 0
        for i in range(0, len(rows), batch_size):
            chunk = rows[i : i + batch_size]
            self.run_write(query, {param_name: chunk})
            total += len(chunk)
        return total

    # -- MERGE helpers (prevent duplicates) ------------------------------------

    def merge_node(
        self,
        label: str,
        match_props: dict,
        extra_on_create: dict | None = None,
        extra_on_match: dict | None = None,
        match_key: str = "name",  # Defaults to "name" for everyone else
    ) -> None:
        params: dict[str, Any] = {"match_props": match_props}
        set_create = "ON CREATE SET n += $on_create" if extra_on_create else ""
        set_match = "ON MATCH SET n += $on_match" if extra_on_match else ""
        
        if extra_on_create: params["on_create"] = extra_on_create
        if extra_on_match: params["on_match"] = extra_on_match

        # This line is the magic: it uses the match_key dynamically
        query = f"""
        MERGE (n:{label} {{{match_key}: $match_props.{match_key}}})
        {set_create}
        {set_match}
        SET n += $match_props
        """
        self.run_write(query, params)

        
    def merge_edge(
        self,
        from_label: str,
        from_props: dict,
        to_label: str,
        to_props: dict,
        rel_type: str,
        rel_props: dict | None = None,
        from_key: str = "name", # Defaults to "name"
        to_key: str = "name"    # Defaults to "name"
    ) -> None:
        params: dict[str, Any] = {
            "from_props": from_props,
            "to_props": to_props,
            "rel_props": rel_props or {}
        }

        # Build the query using the keys provided (or the defaults)
        query = f"""
        MERGE (a:{from_label} {{{from_key}: $from_props.{from_key}}})
        SET a += $from_props
        MERGE (b:{to_label} {{{to_key}: $to_props.{to_key}}})
        SET b += $to_props
        MERGE (a)-[r:{rel_type}]->(b)
        SET r += $rel_props
        """
        self.run_write(query, params)
    
    '''
    def merge_node(
        self,
        label: str,
        match_props: dict,
        extra_on_create: dict | None = None,
        extra_on_match: dict | None = None,
    ) -> None:
        """MERGE a node by label + match properties.

        Uses MERGE to avoid duplicates.  `extra_on_create` is set only when
        the node is first created; `extra_on_match` is set on subsequent hits.
        """
        set_create = ""
        set_match = ""
        params: dict[str, Any] = {"match_props": match_props}

        if extra_on_create:
            set_create = "ON CREATE SET n += $on_create"
            params["on_create"] = extra_on_create
        if extra_on_match:
            set_match = "ON MATCH SET n += $on_match"
            params["on_match"] = extra_on_match

        query = f"""
        MERGE (n:{label} {{name: $match_props.name}})
        {set_create}
        {set_match}
        SET n += $match_props
        """
        self.run_write(query, params)

    def merge_edge(
        self,
        from_label: str,
        from_props: dict,
        to_label: str,
        to_props: dict,
        rel_type: str,
        rel_props: dict | None = None,
    ) -> None:
        """MERGE an edge between two nodes (nodes are also MERGEd).

        This is safe to call multiple times — it will not create duplicate
        nodes or edges.
        """
        params: dict[str, Any] = {
            "from_props": from_props,
            "to_props": to_props,
        }
        rel_clause = ""
        if rel_props:
            rel_clause = "SET r += $rel_props"
            params["rel_props"] = rel_props

        query = f"""
        MERGE (a:{from_label} {{name: $from_props.name}})
        SET a += $from_props
        MERGE (b:{to_label} {{name: $to_props.name}})
        SET b += $to_props
        MERGE (a)-[r:{rel_type}]->(b)
        {rel_clause}
        """
        self.run_write(query, params)

    '''
    # -- property update helpers -----------------------------------------------

    def set_node_properties(
        self,
        label: str,
        match_props: dict,
        set_props: dict,
    ) -> None:
        """MATCH a node and SET additional properties on it.

        Unlike merge_node, this does NOT create the node if it doesn't exist.
        Useful for adding review metadata (archived, archive_reason, etc.).
        """
        query = f"""
        MATCH (n:{label} {{name: $match_props.name}})
        SET n += $set_props
        """
        self.run_write(query, {"match_props": match_props, "set_props": set_props})

    def set_edge_properties(
        self,
        from_label: str,
        from_name: str,
        to_label: str,
        to_name: str,
        rel_type: str,
        set_props: dict,
    ) -> None:
        """MATCH an edge and SET additional properties on it."""
        query = f"""
        MATCH (a:{from_label} {{name: $from_name}})-[r:{rel_type}]->(b:{to_label} {{name: $to_name}})
        SET r += $set_props
        """
        self.run_write(query, {
            "from_name": from_name,
            "to_name": to_name,
            "set_props": set_props,
        })

    # -- convenience -----------------------------------------------------------

    def count_nodes(self, label: str | None = None) -> int:
        """Return total node count, optionally filtered by label."""
        if label:
            result = self.run(f"MATCH (n:{label}) RETURN count(n) AS c")
        else:
            result = self.run("MATCH (n) RETURN count(n) AS c")
        return result[0]["c"] if result else 0

    def count_edges(self, rel_type: str | None = None) -> int:
        """Return total edge count, optionally filtered by type."""
        if rel_type:
            result = self.run(
                f"MATCH ()-[r:{rel_type}]->() RETURN count(r) AS c"
            )
        else:
            result = self.run("MATCH ()-[r]->() RETURN count(r) AS c")
        return result[0]["c"] if result else 0

    def clear_all(self, confirm: bool = False) -> None:
        """Delete everything in the database. Requires confirm=True."""
        if not confirm:
            raise ValueError("Pass confirm=True to delete all data")
        self.run_write("MATCH (n) DETACH DELETE n")
