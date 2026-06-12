"""Phase 1 — Extraction Agent.

Parses historical medical text passages and extracts:
  - Source nodes (remedy organisms/substances)
  - Traditional_Malady nodes (historical symptom/disease descriptions)
  - Preparation_Method nodes
  - TREATS_TRADITIONALLY edges (Source -> Traditional_Malady)
  - PREPARED_AS edges (Source -> Preparation_Method)

Each extraction includes an evidence span and LLM-assessed confidence score.

Uses Google Gemini for structured JSON extraction from historical texts.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from google import genai
from google.genai import types

from src.agents.base import BaseAgent
from src.config import GEMINI_API_KEY, GEMINI_MODEL
from src.graph.client import GraphClient

logger = logging.getLogger(__name__)

# JSON schema describing the expected extraction output from Gemini
EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "sources": {
            "type": "array",
            "description": "Remedy organisms or substances mentioned in the text",
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Canonical organism or substance name (use the most common English or Latin botanical name)"
                    },
                    "aliases": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Alternative names, including original-language names from the text"
                    },
                    "evidence_span": {
                        "type": "string",
                        "description": "Exact text span where this source is mentioned"
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence 0.0-1.0 that this is a real remedy source"
                    }
                },
                "required": ["name", "aliases", "evidence_span", "confidence"]
            }
        },
        "maladies": {
            "type": "array",
            "description": "Symptoms or diseases the remedies claim to treat",
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Malady name normalized to a short English phrase"
                    },
                    "description": {
                        "type": "string",
                        "description": "Longer description of symptoms as stated in the text"
                    },
                    "evidence_span": {
                        "type": "string",
                        "description": "Exact text span describing this malady"
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence 0.0-1.0 that this is a real medical condition"
                    }
                },
                "required": ["name", "description", "evidence_span", "confidence"]
            }
        },
        "preparations": {
            "type": "array",
            "description": "How remedies are prepared or administered",
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Preparation method (e.g., decoction, powder, poultice, tincture)"
                    },
                    "route": {
                        "type": "string",
                        "description": "Administration route if stated (oral, topical, inhalation, etc.)"
                    },
                    "evidence_span": {
                        "type": "string",
                        "description": "Exact text span describing the preparation"
                    }
                },
                "required": ["name", "route", "evidence_span"]
            }
        },
        "relationships": {
            "type": "array",
            "description": "Which source treats which malady, optionally with a preparation method",
            "items": {
                "type": "object",
                "properties": {
                    "source_name": {
                        "type": "string",
                        "description": "Name of the remedy source (must match a name in sources)"
                    },
                    "malady_name": {
                        "type": "string",
                        "description": "Name of the malady (must match a name in maladies)"
                    },
                    "preparation_name": {
                        "type": "string",
                        "description": "Name of the preparation method if specified (must match a name in preparations), or empty string"
                    },
                    "evidence_span": {
                        "type": "string",
                        "description": "Text span supporting this specific relationship"
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence 0.0-1.0 that this source-malady relationship is asserted in the text"
                    }
                },
                "required": ["source_name", "malady_name", "preparation_name", "evidence_span", "confidence"]
            }
        }
    },
    "required": ["sources", "maladies", "preparations", "relationships"]
}


SYSTEM_PROMPT = """\
You are a specialist in historical and traditional medicine text analysis. Your task \
is to extract structured biomedical entities and relationships from historical medical \
text passages.

IMPORTANT GUIDELINES:
- Extract ONLY what is explicitly stated or strongly implied in the text.
- Do NOT hallucinate entities or relationships not supported by the passage.
- Use the most common English name or Latin botanical name as the canonical source name.
- Keep original-language names (Chinese, Arabic, Latin, etc.) as aliases.
- For maladies, normalize to a short English description (e.g., "intermittent fever" not "ague").
- Confidence scores should reflect how clearly the text supports the extraction:
  - 0.9-1.0: Explicitly and unambiguously stated
  - 0.7-0.89: Clearly implied with strong textual support
  - 0.5-0.69: Reasonably inferred but some ambiguity
  - 0.3-0.49: Weak or indirect evidence
  - Below 0.3: Do not extract — too speculative
- Evidence spans should be exact quotes from the passage (or close paraphrases if the \
  original is in a non-English language).
- If a passage contains no extractable medical entities, return empty arrays.\
"""


class ExtractionAgent(BaseAgent):

    def __init__(self, client: GraphClient, **kwargs: Any):
        super().__init__(client, **kwargs)
        api_key = kwargs.get("gemini_api_key") or GEMINI_API_KEY
        if not api_key:
            raise ValueError(
                "Gemini API key required. Set GEMINI_API_KEY in .env or pass gemini_api_key=..."
            )
        self._gemini = genai.Client(api_key=api_key)
        self._model = kwargs.get("gemini_model") or GEMINI_MODEL

    @property
    def name(self) -> str:
        return "ExtractionAgent"

    def run(self, passages: list[dict], **kwargs: Any) -> dict:
        """Extract entities and relations from historical text passages.

        Args:
            passages: List of dicts with keys:
                - "text": the raw passage text
                - "source_document": title of the historical text
                - "chunk_id": identifier for this chunk

        Returns:
            Summary dict with counts and errors.
        """
        summary = {
            "nodes_created": 0,
            "edges_created": 0,
            "passages_processed": 0,
            "passages_failed": 0,
            "errors": [],
        }

        for i, passage in enumerate(passages):
            self._log_progress(
                f"Processing passage {i + 1}/{len(passages)}: {passage.get('chunk_id', 'unknown')}"
            )
            try:
                extraction = self._extract_from_passage(
                    passage["text"], passage["source_document"]
                )
                counts = self._write_to_graph(extraction, passage["source_document"])
                summary["nodes_created"] += counts["nodes"]
                summary["edges_created"] += counts["edges"]
                summary["passages_processed"] += 1
            except Exception as e:
                msg = f"Passage {passage.get('chunk_id', i)}: {e}"
                logger.error(msg)
                summary["errors"].append(msg)
                summary["passages_failed"] += 1

        self._log_progress(
            f"Done — {summary['nodes_created']} nodes, "
            f"{summary['edges_created']} edges, "
            f"{summary['passages_failed']} failures"
        )
        return summary

    # ------------------------------------------------------------------
    # LLM interaction
    # ------------------------------------------------------------------

    def _extract_from_passage(self, text: str, source_doc: str) -> dict:
        """Call Gemini to extract structured entities from a passage."""
        prompt = self._build_extraction_prompt(text, source_doc)

        response = self._gemini.models.generate_content(
            model=self._model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=EXTRACTION_SCHEMA,
            ),
        )

        return self._parse_llm_response(response.text)

    def _build_extraction_prompt(self, passage: str, source_doc: str) -> str:
        """Build the user prompt for structured entity extraction."""
        return f"""\
Analyze the following historical medical text passage from "{source_doc}" and extract \
all remedy sources, maladies/symptoms, preparation methods, and their relationships.

--- BEGIN PASSAGE ---
{passage}
--- END PASSAGE ---

Extract all entities and relationships as structured JSON following the schema provided. \
Remember:
- source names should use the most common English or Latin botanical name
- keep original-language names as aliases
- normalize malady names to short English descriptions
- include exact text evidence spans for every extraction
- assign confidence scores based on how clearly the text supports each extraction
- link sources to maladies via the relationships array
- if a preparation method is mentioned for a source-malady pair, include it"""

    def _parse_llm_response(self, response_text: str) -> dict:
        """Parse structured JSON from LLM response."""
        try:
            data = json.loads(response_text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse Gemini JSON response: {e}") from e

        # Validate expected top-level keys
        for key in ("sources", "maladies", "preparations", "relationships"):
            if key not in data:
                data[key] = []

        return data

    # ------------------------------------------------------------------
    # Graph writes
    # ------------------------------------------------------------------

    def _write_to_graph(self, extraction: dict, source_document: str) -> dict:
        """Write extracted entities and relationships to Neo4j.

        Returns dict with node and edge counts.
        """
        node_count = 0
        edge_count = 0

        # 1. MERGE Source nodes
        for src in extraction.get("sources", []):
            self.client.merge_node(
                "Source",
                {"name": src["name"]},
                extra_on_create={
                    "aliases": src.get("aliases", []),
                    "source_document": source_document,
                    "evidence_span": src.get("evidence_span", ""),
                    "created_by": "extraction_agent",
                },
            )
            node_count += 1

        # 2. MERGE Traditional_Malady nodes
        for mal in extraction.get("maladies", []):
            self.client.merge_node(
                "Traditional_Malady",
                {"name": mal["name"]},
                extra_on_create={
                    "description": mal.get("description", ""),
                    "source_document": source_document,
                    "evidence_span": mal.get("evidence_span", ""),
                    "created_by": "extraction_agent",
                },
            )
            node_count += 1

        # 3. MERGE Preparation_Method nodes
        for prep in extraction.get("preparations", []):
            self.client.merge_node(
                "Preparation_Method",
                {"name": prep["name"]},
                extra_on_create={
                    "route": prep.get("route", ""),
                    "evidence_span": prep.get("evidence_span", ""),
                    "created_by": "extraction_agent",
                },
            )
            node_count += 1

        # 4. Create edges from relationships
        for rel in extraction.get("relationships", []):
            source_name = rel["source_name"]
            malady_name = rel["malady_name"]
            prep_name = rel.get("preparation_name", "")
            confidence = rel.get("confidence", 0.5)
            evidence = rel.get("evidence_span", "")

            # TREATS_TRADITIONALLY edge: Source -> Traditional_Malady
            self.client.merge_edge(
                "Source", {"name": source_name},
                "Traditional_Malady", {"name": malady_name},
                "TREATS_TRADITIONALLY",
                {
                    "confidence_score": confidence,
                    "evidence_span": evidence,
                    "created_by": "extraction_agent",
                    "reviewed": False,
                },
            )
            edge_count += 1

            # PREPARED_AS edge: Source -> Preparation_Method (if specified)
            if prep_name:
                self.client.merge_edge(
                    "Source", {"name": source_name},
                    "Preparation_Method", {"name": prep_name},
                    "PREPARED_AS",
                    {
                        "confidence_score": confidence,
                        "evidence_span": evidence,
                        "created_by": "extraction_agent",
                        "reviewed": False,
                    },
                )
                edge_count += 1

        return {"nodes": node_count, "edges": edge_count}
