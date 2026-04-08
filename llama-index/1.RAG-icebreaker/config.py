"""Configuration settings for the Icebreaker Bot."""

import os
from dotenv import load_dotenv
load_dotenv(override=True)

# =========================
# OpenAI settings
# =========================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")  # Set via environment variable preferably
OPENAI_BASE_URL = "https://api.openai.com/v1"

# =========================
# Model settings
# =========================
# Chat / LLM model
LLM_MODEL_ID = "gpt-4o-mini"
# Alternatives:
# - "gpt-4o-mini"  
# - "gpt-4.1"
# - "gpt-4o"
# - "gpt-3.5-turbo"

# Embedding model
EMBEDDING_MODEL_ID = "text-embedding-3-small"
# Alternative (higher quality):
# "text-embedding-3-large"

# =========================
# ProxyCurl API settings
# =========================
PROXYCURL_API_KEY = ""  # Replace with your API key

# =========================
# Mock data URL
# =========================
MOCK_DATA_URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ZRe59Y_NJyn3hZgnF1iFYA/linkedin-profile-data.json"

# =========================
# Retrieval / Query settings
# =========================
SIMILARITY_TOP_K = 5

# OpenAI generation settings
TEMPERATURE = 0.0
MAX_NEW_TOKENS = 500
TOP_P = 1.0

# =========================
# Chunking settings
# =========================
CHUNK_SIZE = 500

# =========================
# Prompt templates
# =========================
INITIAL_FACTS_TEMPLATE = """
You are an AI assistant that provides detailed answers based on the provided context.

Context information is below:

{context_str}

Based on the context provided, list 3 interesting facts about this person's
career or education.

Answer in detail, using only the information provided in the context.
"""

USER_QUESTION_TEMPLATE = """
You are an AI assistant that provides detailed answers to questions based on the
provided context.

Context information is below:

{context_str}

Question: {query_str}

Answer in full detail, using only the information provided in the context.
If the answer is not available in the context, say:
"I don't know. The information is not available on the LinkedIn page."
"""
