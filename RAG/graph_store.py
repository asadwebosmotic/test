"""
Graph Store Module - Memgraph operations using gqlalchemy

This module handles storing document, chunk, and entity nodes in Memgraph
for GraphRAG functionality.
"""
import uuid
import logging
from typing import Optional, List, Tuple, Dict
from gqlalchemy import Memgraph

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

# === Configure logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Initialize Memgraph connection ===
memgraph = Memgraph(
    host=settings.MEMGRAPH_HOST,
    port=settings.MEMGRAPH_PORT
)


def store_document_in_memgraph(filename: str, user_id: str) -> str:
    """
    Creates a Document node in Memgraph.
    
    Args:
        filename (str): Name of the PDF file
        user_id (str): User identifier for data isolation
        
    Returns:
        str: The generated document_id
    """
    document_id = str(uuid.uuid4())
    
    # Create Document node with Cypher query
    query = """
    CREATE (d:Document {
        id: $doc_id,
        filename: $filename,
        user_id: $user_id,
        created_at: timestamp()
    })
    RETURN d.id AS document_id
    """
    
    try:
        result = memgraph.execute_and_fetch(
            query, 
            {"doc_id": document_id, "filename": filename, "user_id": user_id}
        )
        result_list = list(result)
        if result_list:
            logger.info(f"✅ Created Document node: {document_id} for file {filename}")
            return document_id
    except Exception as e:
        logger.error(f"❌ Failed to create Document node: {e}")
        raise
    
    return document_id


def store_entity_in_memgraph(entity_name: str, entity_label: str) -> str:
    """
    Creates or retrieves an Entity node in Memgraph.
    Uses MERGE to avoid duplicate entities.
    
    Args:
        entity_name (str): The entity text
        entity_label (str): The entity type/label (PERSON, ORG, etc.)
        
    Returns:
        str: The entity_id
    """
    entity_id = str(uuid.uuid4())
    
    # MERGE to avoid duplicates - match on name AND label
    query = """
    MERGE (e:Entity {name: $name, label: $label})
    ON CREATE SET e.id = $entity_id, e.created_at = timestamp()
    RETURN e.id AS entity_id
    """
    
    try:
        result = memgraph.execute_and_fetch(
            query,
            {"name": entity_name, "label": entity_label, "entity_id": entity_id}
        )
        result_list = list(result)
        if result_list:
            return result_list[0].get("entity_id", entity_id)
    except Exception as e:
        logger.warning(f"⚠️ Failed to create Entity node for {entity_name}: {e}")
    
    return entity_id


def store_chunk_in_memgraph(doc_id: str, chunk: dict) -> str:
    """
    Creates a Chunk node and links it to a Document via HAS_CHUNK relationship.
    Also creates Entity nodes and HAS_ENTITY relationships if entities are present.
    
    Args:
        doc_id (str): The parent Document ID
        chunk (dict): Chunk data containing:
            - page_content: The text content
            - metadata: dict with page_number, source, type, entities, phrases
            
    Returns:
        str: The generated chunk_id
    """
    chunk_id = str(uuid.uuid4())
    text = chunk.get("page_content", "")
    metadata = chunk.get("metadata", {})
    page = metadata.get("page_number", 1)
    chunk_type = metadata.get("type", "text")
    source = metadata.get("source", "unknown")
    entities = metadata.get("entities", [])  # List of (name, label) tuples
    phrases = metadata.get("phrases", [])
    
    # Create Chunk node and relationship with Cypher query
    query = """
    MATCH (d:Document {id: $doc_id})
    CREATE (c:Chunk {
        id: $chunk_id,
        text: $text,
        page: $page,
        type: $chunk_type,
        source: $source,
        phrases: $phrases
    })
    CREATE (d)-[:HAS_CHUNK]->(c)
    RETURN c.id AS chunk_id
    """
    
    try:
        result = memgraph.execute_and_fetch(
            query, 
            {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "text": text,
                "page": page,
                "chunk_type": chunk_type,
                "source": source,
                "phrases": phrases
            }
        )
        result_list = list(result)
        if result_list:
            logger.info(f"✅ Created Chunk node: {chunk_id[:8]}... linked to Document {doc_id[:8]}...")
            
            # Create Entity nodes and relationships
            if entities:
                for entity_name, entity_label in entities:
                    if entity_name and entity_label:
                        store_chunk_entity_relationship(chunk_id, entity_name, entity_label)
                        
    except Exception as e:
        logger.error(f"❌ Failed to create Chunk node: {e}")
        raise
    
    return chunk_id


def store_chunk_entity_relationship(chunk_id: str, entity_name: str, entity_label: str):
    """
    Creates an Entity node (if not exists) and links it to a Chunk via HAS_ENTITY.
    
    Args:
        chunk_id (str): The Chunk ID
        entity_name (str): Entity text
        entity_label (str): Entity type (PERSON, ORG, GPE, etc.)
    """
    # MERGE entity then CREATE relationship
    query = """
    MATCH (c:Chunk {id: $chunk_id})
    MERGE (e:Entity {name: $entity_name, label: $entity_label})
    ON CREATE SET e.id = $entity_id, e.created_at = timestamp()
    MERGE (c)-[:HAS_ENTITY]->(e)
    """
    
    entity_id = str(uuid.uuid4())
    
    try:
        memgraph.execute(
            query,
            {
                "chunk_id": chunk_id,
                "entity_name": entity_name,
                "entity_label": entity_label,
                "entity_id": entity_id
            }
        )
        logger.debug(f"🔗 Linked entity '{entity_name}' ({entity_label}) to chunk {chunk_id[:8]}...")
    except Exception as e:
        logger.warning(f"⚠️ Failed to create entity relationship: {e}")


# def store_chunks_batch_in_memgraph(doc_id: str, chunks: list) -> int:
#     """
#     Store multiple chunks in Memgraph in a batch operation.
    
#     Args:
#         doc_id (str): The parent Document ID
#         chunks (list): List of chunk dictionaries
        
#     Returns:
#         int: Number of chunks stored
#     """
#     stored_count = 0
#     entity_count = 0
    
#     for chunk in chunks:
#         try:
#             store_chunk_in_memgraph(doc_id, chunk)
#             stored_count += 1
            
#             # Count entities
#             entities = chunk.get("metadata", {}).get("entities", [])
#             entity_count += len(entities)
            
#         except Exception as e:
#             logger.warning(f"⚠️ Failed to store chunk: {e}")
#             continue
    
#     logger.info(f"📊 Stored {stored_count}/{len(chunks)} chunks with {entity_count} entity relationships in Memgraph")
#     return stored_count

def store_chunks_batch_in_memgraph(document_id: str, chunks: List[Dict]) -> int:
    """
    Stores all chunks + entity nodes + co-occurrence edges in Memgraph.
    
    Each chunk may have:
    - chunk["page_content"]
    - chunk["metadata"]["page_number"]
    - chunk["metadata"]["entities"] (list)
    - chunk["metadata"]["phrases"] (list)
    """

    stored_count = 0

    for chunk in chunks:
        chunk_id = str(uuid.uuid4())

        text = chunk.get("page_content", "")
        meta = chunk.get("metadata", {})
        page = meta.get("page_number", 1)
        entities = meta.get("entities", [])

        # === 1. Create the Chunk node ===
        memgraph.execute(
            """
            MATCH (d:Document {id: $doc_id})
            CREATE (c:Chunk {
                id: $chunk_id,
                text: $text,
                page: $page
            })
            CREATE (d)-[:HAS_CHUNK]->(c)
            """,
            {
                "doc_id": document_id,
                "chunk_id": chunk_id,
                "text": text,
                "page": page,
            }
        )
        stored_count += 1

        # === 2. Create Entity nodes + MENTIONS edges ===
        for ent in entities:
            memgraph.execute(
                """
                MERGE (e:Entity {name: $name})
                WITH e
                MATCH (c:Chunk {id: $chunk_id})
                MERGE (c)-[:MENTIONS]->(e)
                """,
                {"name": ent, "chunk_id": chunk_id}
            )

        # === 3. Create CO-OCCURS edges between every pair of entities ===
        if len(entities) > 1:
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    e1 = entities[i]
                    e2 = entities[j]

                    memgraph.execute(
                        """
                        MERGE (a:Entity {name: $e1})
                        MERGE (b:Entity {name: $e2})
                        MERGE (a)-[r:CO_OCCURS]-(b)
                        ON CREATE SET r.count = 1
                        ON MATCH SET r.count = r.count + 1
                        """,
                        {"e1": e1, "e2": e2}
                    )

    logger.info(f"🕸️ Stored {stored_count} chunks + entity graph in Memgraph.")
    return stored_count