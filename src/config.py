"""Central configuration loaded from environment variables."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

# Neo4j
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# LLM
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")
GEMINI_MODEL_PRO = os.getenv("GEMINI_MODEL_PRO", "gemini-3.1-pro-preview")
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "")
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")


def make_gemini_client():
    """Return a configured google-genai Client, preferring Vertex AI Express API key."""
    from google import genai
    if GOOGLE_API_KEY:
        return genai.Client(vertexai=True, api_key=GOOGLE_API_KEY)
    if GOOGLE_CLOUD_PROJECT:
        return genai.Client(vertexai=True, project=GOOGLE_CLOUD_PROJECT, location=GOOGLE_CLOUD_LOCATION)
    return genai.Client(api_key=GEMINI_API_KEY)

# Ontology API endpoints (all free, no auth required)
ICD10_API_BASE = "https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search"
MESH_LOOKUP_BASE = "https://id.nlm.nih.gov/mesh/lookup/descriptor"
OLS4_API_BASE = "https://www.ebi.ac.uk/ols4/api/search"
ONTOLOGY_API_RATE_LIMIT = int(os.getenv("ONTOLOGY_API_RATE_LIMIT", "20"))

# External APIs
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")
S2_API_KEY = os.getenv("S2_API_KEY", "")
