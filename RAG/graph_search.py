"""
Graph Search Module - Memgraph Graph Expansion

This module handles expanding context using graph relationships in Memgraph
for GraphRAG functionality, including entity-based retrieval.
"""
import logging
from typing import List, Optional
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


def expand_graph_context(chunk_text: str, limit: int = 10, query_entities: List[str] = None) -> List[str]:
    """
    Expand context by finding related chunks through graph relationships.
    
    Uses entity-based relevance: only returns sibling chunks that share
    at least one entity with the source chunk OR with query entities.
    Falls back to page-proximity if no entity matches found.
    
    Args:
        chunk_text (str): The text of the chunk to expand from
        limit (int): Maximum number of related chunks to return
        query_entities (List[str]): Optional list of entities from the user query
        
    Returns:
        List[str]: List of related chunk texts (filtered by relevance)
    """
    logger.info(f"🕸️ Expanding graph context for chunk: '{chunk_text[:50]}...'")
    
    # Strategy 1: Find chunks that share entities with the source chunk
    # This ensures relevance by entity co-occurrence
    query_entity_based = """
    MATCH (c:Chunk {text: $text})-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(neighbors:Chunk)
    WHERE neighbors.text <> $text
    RETURN DISTINCT neighbors.text AS related, neighbors.page AS page
    LIMIT $limit
    """
    
    try:
        result = memgraph.execute_and_fetch(
            query_entity_based,
            {"text": chunk_text, "limit": limit}
        )
        
        related_texts = []
        for record in result:
            text = record.get("related")
            if text:
                related_texts.append(text)
        
        # If we found entity-related chunks, return them
        if related_texts:
            logger.info(f"✅ Found {len(related_texts)} entity-related chunks from graph")
            return related_texts
        
        # Strategy 2: If query_entities provided, find chunks matching those entities
        if query_entities:
            logger.info(f"🔍 Trying query entity matching with {len(query_entities)} entities...")
            for entity in query_entities:
                entity_query = """
                MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
                WHERE toLower(e.name) CONTAINS toLower($entity)
                   OR toLower($entity) CONTAINS toLower(e.name)
                RETURN DISTINCT c.text AS related, e.name AS matched
                LIMIT $limit
                """
                entity_result = memgraph.execute_and_fetch(
                    entity_query,
                    {"entity": entity, "limit": limit}
                )
                for record in entity_result:
                    text = record.get("related")
                    if text and text != chunk_text:
                        related_texts.append(text)
            
            if related_texts:
                # Deduplicate
                related_texts = list(set(related_texts))[:limit]
                logger.info(f"✅ Found {len(related_texts)} query-entity-matched chunks")
                return related_texts
        
        # Strategy 3: Fallback - get adjacent page chunks only (same document, nearby pages)
        logger.info("⚠️ No entity matches, falling back to page-proximity...")
        proximity_query = """
        MATCH (c:Chunk {text: $text})<-[:HAS_CHUNK]-(d:Document)
        MATCH (d)-[:HAS_CHUNK]->(neighbors:Chunk)
        WHERE neighbors.text <> $text 
          AND abs(neighbors.page - c.page) <= 1
        RETURN neighbors.text AS related
        LIMIT $limit
        """
        
        prox_result = memgraph.execute_and_fetch(
            proximity_query,
            {"text": chunk_text, "limit": limit}
        )
        
        for record in prox_result:
            text = record.get("related")
            if text:
                related_texts.append(text)
        
        logger.info(f"✅ Found {len(related_texts)} proximity-based chunks")
        return related_texts
        
    except Exception as e:
        logger.error(f"❌ Graph expansion failed: {e}")
        return []


def get_chunks_by_entity(entity_name: str, limit: int = 10) -> List[str]:
    """
    Get chunks that contain a specific entity.
    Uses case-insensitive fuzzy matching to find entities.
    
    Args:
        entity_name (str): The entity name/text to search for
        limit (int): Maximum number of chunks to return
        
    Returns:
        List[str]: List of chunk texts containing the entity
    """
    logger.info(f"🔗 Getting chunks for entity: '{entity_name}'")
    
    # Use case-insensitive fuzzy matching with CONTAINS
    # This allows "anil kumar tandel" to match "ANILKUMAR V TANDEL"
    query = """
    MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
    WHERE toLower(e.name) CONTAINS toLower($entity)
       OR toLower($entity) CONTAINS toLower(e.name)
    RETURN DISTINCT c.text AS text, e.name AS matched_entity
    LIMIT $limit
    """
    
    try:
        result = memgraph.execute_and_fetch(
            query,
            {"entity": entity_name, "limit": limit}
        )
        
        chunk_texts = []
        matched_entities = set()
        for record in result:
            text = record.get("text")
            matched = record.get("matched_entity")
            if text:
                chunk_texts.append(text)
            if matched:
                matched_entities.add(matched)
        
        if matched_entities:
            logger.info(f"✅ Found {len(chunk_texts)} chunks for entity '{entity_name}' (matched: {list(matched_entities)[:3]})")
        else:
            logger.info(f"⚠️ No entity match found for '{entity_name}'")
        return chunk_texts
        
    except Exception as e:
        logger.error(f"❌ Entity chunk retrieval failed: {e}")
        return []


def get_chunks_by_entities(entity_names: List[str], limit_per_entity: int = 5) -> List[str]:
    """
    Get chunks for multiple entities and aggregate results.
    
    Args:
        entity_names (List[str]): List of entity names to search
        limit_per_entity (int): Maximum chunks per entity
        
    Returns:
        List[str]: Deduplicated list of chunk texts
    """
    if not entity_names:
        return []
    
    all_chunks = set()
    
    for entity_name in entity_names:
        chunks = get_chunks_by_entity(entity_name, limit=limit_per_entity)
        all_chunks.update(chunks)
    
    logger.info(f"🔗 Total unique chunks from {len(entity_names)} entities: {len(all_chunks)}")
    return list(all_chunks)


def expand_graph_context_by_document(document_id: str, limit: int = 10) -> List[str]:
    """
    Expand context by getting all chunks from a specific document.
    
    Args:
        document_id (str): The document ID
        limit (int): Maximum number of chunks to return
        
    Returns:
        List[str]: List of chunk texts from the document
    """
    logger.info(f"🕸️ Getting chunks from document: {document_id[:8]}...")
    
    query = """
    MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c:Chunk)
    RETURN c.text AS text, c.page AS page, c.type AS type
    ORDER BY c.page
    LIMIT $limit
    """
    
    try:
        result = memgraph.execute_and_fetch(
            query,
            {"doc_id": document_id, "limit": limit}
        )
        
        chunk_texts = []
        for record in result:
            text = record.get("text")
            if text:
                chunk_texts.append(text)
        
        logger.info(f"✅ Retrieved {len(chunk_texts)} chunks from document")
        return chunk_texts
        
    except Exception as e:
        logger.error(f"❌ Document chunk retrieval failed: {e}")
        return []


def expand_multi_chunk_context(chunk_texts: List[str], limit_per_chunk: int = 5, query_entities: List[str] = None) -> List[str]:
    """
    Expand context for multiple chunks and aggregate results.
    Uses entity-based relevance filtering.
    
    Args:
        chunk_texts (List[str]): List of chunk texts to expand
        limit_per_chunk (int): Maximum related chunks per input chunk
        query_entities (List[str]): Optional entities from user query for relevance
        
    Returns:
        List[str]: Deduplicated list of related chunk texts
    """
    all_related = set()
    
    for chunk_text in chunk_texts:
        related = expand_graph_context(chunk_text, limit=limit_per_chunk, query_entities=query_entities)
        all_related.update(related)
    
    # Remove original chunks from results
    all_related -= set(chunk_texts)
    
    logger.info(f"🕸️ Total unique related chunks: {len(all_related)}")
    return list(all_related)


def get_related_entities(entity_name: str, limit: int = 10) -> List[dict]:
    """
    Find entities that co-occur in the same chunks as the given entity.
    
    Args:
        entity_name (str): The entity to find co-occurring entities for
        limit (int): Maximum number of related entities
        
    Returns:
        List[dict]: List of related entity info {name, label}
    """
    logger.info(f"🔗 Finding entities related to: '{entity_name}'")
    
    query = """
    MATCH (e1:Entity {name: $entity})<-[:HAS_ENTITY]-(c:Chunk)
    MATCH (c)-[:HAS_ENTITY]->(e2:Entity)
    WHERE e2.name <> $entity
    RETURN DISTINCT e2.name AS name, e2.label AS label
    LIMIT $limit
    """
    
    try:
        result = memgraph.execute_and_fetch(
            query,
            {"entity": entity_name, "limit": limit}
        )
        
        related_entities = []
        for record in result:
            related_entities.append({
                "name": record.get("name"),
                "label": record.get("label")
            })
        
        logger.info(f"✅ Found {len(related_entities)} related entities")
        return related_entities
        
    except Exception as e:
        logger.error(f"❌ Related entity retrieval failed: {e}")
        return []
