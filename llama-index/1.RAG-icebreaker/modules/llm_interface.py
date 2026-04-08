"""Module for interfacing with OpenAI LLMs using LlamaIndex."""

import logging
from typing import Optional

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

import config

logger = logging.getLogger(__name__)


def create_openai_embedding() -> OpenAIEmbedding:
    """
    Creates an OpenAI Embedding model for vector representation.

    Returns:
        OpenAIEmbedding model.
    """
    embedding = OpenAIEmbedding(
        model=config.EMBEDDING_MODEL_ID,
        api_key=config.OPENAI_API_KEY,
    )

    logger.info(
        f"Created OpenAI Embedding model: {config.EMBEDDING_MODEL_ID}"
    )
    return embedding


def create_openai_llm(
    temperature: float = config.TEMPERATURE,
    max_new_tokens: int = config.MAX_NEW_TOKENS,
) -> OpenAI:
    """
    Creates an OpenAI LLM for generating responses.

    Args:
        temperature: Controls randomness in generation (0.0 to 1.0).
        max_new_tokens: Maximum number of tokens to generate.

    Returns:
        OpenAI LLM model.
    """
    llm = OpenAI(
        model=config.LLM_MODEL_ID,
        api_key=config.OPENAI_API_KEY,
        temperature=temperature,
        max_tokens=max_new_tokens,
        top_p=config.TOP_P,
    )

    logger.info(f"Created OpenAI LLM model: {config.LLM_MODEL_ID}")
    return llm


def change_llm_model(new_model_id: str) -> None:
    """
    Change the LLM model to use.

    Args:
        new_model_id: New LLM model ID to use.
    """
    config.LLM_MODEL_ID = new_model_id
    logger.info(f"Changed LLM model to: {new_model_id}")
