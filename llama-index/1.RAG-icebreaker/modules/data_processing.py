"""Module for processing LinkedIn profile data."""

import json
import logging
from typing import Dict, List, Any, Optional

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter

from modules.llm_interface import create_openai_embedding
import config

logger = logging.getLogger(__name__)


def split_profile_data(profile_data: Dict[str, Any]) -> List:
    """
    Splits the LinkedIn profile JSON data into nodes.

    Args:
        profile_data: LinkedIn profile data dictionary.

    Returns:
        List of document nodes.
    """
    try:
        # Convert the profile data to a JSON string
        profile_json = json.dumps(profile_data, ensure_ascii=False)

        # Create a Document object
        document = Document(text=profile_json)

        # Split document into nodes(chunks)
        splitter = SentenceSplitter(chunk_size=config.CHUNK_SIZE)
        nodes = splitter.get_nodes_from_documents([document])

        logger.info(f"Created {len(nodes)} nodes from profile data")
        return nodes

    except Exception as e:
        logger.error(f"Error in split_profile_data: {e}", exc_info=True)
        return []


def create_vector_database(nodes: List) -> Optional[VectorStoreIndex]:
    """
    Stores the document chunks (nodes) in a vector database.

    Args:
        nodes: List of document nodes to be indexed.

    Returns:
        VectorStoreIndex or None if indexing fails.
    """
    try:
        # Create OpenAI embedding model
        embedding_model = create_openai_embedding()

        # Build vector index
        index = VectorStoreIndex(
            nodes=nodes,
            embed_model=embedding_model,
            show_progress=False,
        )

        logger.info("Vector database created successfully")
        return index

    except Exception as e:
        logger.error(f"Error in create_vector_database: {e}", exc_info=True)
        return None


def verify_embeddings(index: VectorStoreIndex) -> bool:
    """
    Verify that all nodes have been properly embedded.

    Args:
        index: VectorStoreIndex to verify.

    Returns:
        True if all embeddings are valid, False otherwise.
    """
    try:
        vector_store = index._storage_context.vector_store
        node_ids = list(index.index_struct.nodes_dict.keys())

        missing_embeddings = False

        for node_id in node_ids:
            embedding = vector_store.get(node_id)
            if embedding is None:
                logger.warning(f"Node ID {node_id} has a missing embedding.")
                missing_embeddings = True

        if missing_embeddings:
            logger.warning("Some node embeddings are missing")
            return False

        logger.info("All node embeddings are valid")
        return True

    except Exception as e:
        logger.error(f"Error in verify_embeddings: {e}", exc_info=True)
        return False
