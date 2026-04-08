"""Module for querying indexed LinkedIn profile data."""

import logging
from typing import Any

from llama_index.core import VectorStoreIndex, PromptTemplate

from modules.llm_interface import create_openai_llm
import config

logger = logging.getLogger(__name__)


def generate_initial_facts(index: VectorStoreIndex) -> str:
    """
    Generates interesting facts about the person's career or education.

    Args:
        index: VectorStoreIndex containing the LinkedIn profile data.

    Returns:
        String containing interesting facts about the person.
    """
    try:
        # Create OpenAI LLM
        llm = create_openai_llm(
            temperature=0.0,
            max_new_tokens=config.MAX_NEW_TOKENS,
        )

        # Prompt template
        facts_prompt = PromptTemplate(
            template=config.INITIAL_FACTS_TEMPLATE
        )

        # Query engine
        query_engine = index.as_query_engine(
            llm=llm,
            similarity_top_k=config.SIMILARITY_TOP_K,
            streaming=False,
            text_qa_template=facts_prompt, # Based on query, top 5 nodes(chunks) will be fetched and will be passed in Prompt Template with variable name as "context_str".
            # query will not passed into prompt template because there is no "query_str" in prompt template.
        )

        # Execute query
        query = "Provide three interesting facts about this person's career or education."
        response = query_engine.query(query)

        return response.response

    except Exception as e:
        logger.error(
            f"Error in generate_initial_facts: {e}", exc_info=True
        )
        return "Failed to generate initial facts."


def answer_user_query(index: VectorStoreIndex, user_query: str) -> Any:
    """
    Answers the user's question using the vector database and the LLM.

    Args:
        index: VectorStoreIndex containing the LinkedIn profile data.
        user_query: The user's question.

    Returns:
        Response object containing the answer.
    """
    try:
        # Create OpenAI LLM
        llm = create_openai_llm(
            temperature=0.0,
            max_new_tokens=250,
        )

        # Prompt template
        question_prompt = PromptTemplate(
            template=config.USER_QUESTION_TEMPLATE
        )

        # (Optional) Retrieve source nodes for logging / debugging
        retriever = index.as_retriever(
            similarity_top_k=config.SIMILARITY_TOP_K
        )
        source_nodes = retriever.retrieve(user_query)

        logger.debug(
            f"Retrieved {len(source_nodes)} source nodes for query"
        )

        # Query engine
        query_engine = index.as_query_engine(
            llm=llm,
            similarity_top_k=config.SIMILARITY_TOP_K,
            streaming=False,
            text_qa_template=question_prompt, # Based on query, top 5 nodes(chunks) will be fetched and will be passed in Prompt Template with variable name as "context_str".
            # and "query" will passed into prompt template with variable name as "query_str".
        )

        # Execute query
        answer = query_engine.query(user_query)
        return answer

    except Exception as e:
        logger.error(
            f"Error in answer_user_query: {e}", exc_info=True
        )
        return "Failed to get an answer."
