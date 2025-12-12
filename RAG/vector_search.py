"""
Vector Search Module - Qdrant Vector Search

This module handles semantic search using Qdrant for Vector RAG.
Updated for qdrant-client >= 1.7.0 which uses query_points instead of search.
"""
import logging
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

# Import the embedding model from existing module
from RAG.embedding_and_store import embed_model

# === Configure logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Initialize Qdrant client ===
client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)

COLLECTION_NAME = "KnowMe_chunks"

# === Relevance thresholds ===
# Minimum score for a result to be considered relevant (0.0 to 1.0)
# Higher = stricter filtering, fewer but more relevant results
MIN_RELEVANCE_SCORE = 0.80


def search_qdrant(
    query: str, 
    user_id: str = "anonymous", 
    top_k: int = 5,
    min_score: float = MIN_RELEVANCE_SCORE
) -> List[Dict]:
    """
    Search Qdrant for the most relevant chunks based on the query.
    Only returns results above the minimum relevance score.
    
    Args:
        query (str): The search query
        user_id (str): User identifier for filtering results
        top_k (int): Number of results to return (default: 5)
        min_score (float): Minimum relevance score threshold (default: 0.80)
        
    Returns:
        List[Dict]: List of matching chunks with text and metadata (only if score >= min_score)
    """
    logger.info(f"🔍 Searching Qdrant for query: '{query[:50]}...' (user: {user_id}, min_score: {min_score})")
    
    # Embed the query using the same model used for storage
    query_embedding = embed_model.encode(query).tolist()
    
    # Create filter for user_id
    user_filter = Filter(
        must=[
            FieldCondition(
                key="user_id",
                match=MatchValue(value=user_id)
            )
        ]
    )
    
    try:
        # Perform vector search using query_points (qdrant-client >= 1.7.0)
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            query_filter=user_filter,
            limit=top_k,
            with_payload=True
        )
        
        # Format results - only include results above the relevance threshold
        formatted_results = []
        filtered_count = 0
        
        for point in results.points:
            score = point.score
            
            # Filter by minimum relevance score
            if score < min_score:
                filtered_count += 1
                logger.debug(f"⚠️ Filtered out result with score {score:.3f} (below {min_score})")
                continue
                
            formatted_results.append({
                "text": point.payload.get("text", ""),
                "page": point.payload.get("page", 1),
                "source": point.payload.get("source", "unknown"),
                "type": point.payload.get("type", "text"),
                "score": score
            })
        
        logger.info(f"✅ Found {len(formatted_results)} relevant results from Qdrant "
                   f"(filtered out {filtered_count} low-score results)")
        return formatted_results
        
    except Exception as e:
        logger.error(f"❌ Qdrant search failed: {e}")
        return []


def search_qdrant_enhanced(
    query: str, 
    user_id: str = "anonymous", 
    top_k: int = 5,
    entities: Optional[List[str]] = None,
    phrases: Optional[List[str]] = None,
    min_score: float = MIN_RELEVANCE_SCORE
) -> List[Dict]:
    """
    Enhanced search with spaCy-extracted entities and noun phrases.
    Only returns results above the minimum relevance score.
    
    Args:
        query (str): The original search query
        user_id (str): User identifier for filtering
        top_k (int): Number of results to return
        entities (List[str]): Extracted entities from spaCy
        phrases (List[str]): Extracted noun phrases from spaCy
        min_score (float): Minimum relevance score threshold
        
    Returns:
        List[Dict]: List of matching chunks (only if score >= min_score)
    """
    # Build enhanced query by appending entities and phrases
    enhanced_parts = [query]
    
    if entities:
        enhanced_parts.extend(entities)
    if phrases:
        enhanced_parts.extend(phrases)
    
    enhanced_query = " ".join(enhanced_parts)
    
    logger.info(f"🔍 Enhanced search with: entities={entities}, phrases={phrases}")
    
    return search_qdrant(enhanced_query, user_id, top_k, min_score)


def search_qdrant_all_users(query: str, top_k: int = 5) -> List[Dict]:
    """
    Search Qdrant without user filtering (for admin/global queries).
    
    Args:
        query (str): The search query
        top_k (int): Number of results to return
        
    Returns:
        List[Dict]: List of matching chunks with text and metadata
    """
    logger.info(f"🔍 Searching Qdrant (all users) for query: '{query[:50]}...'")
    
    query_embedding = embed_model.encode(query).tolist()
    
    try:
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=top_k,
            with_payload=True
        )
        
        formatted_results = []
        for point in results.points:
            formatted_results.append({
                "text": point.payload.get("text", ""),
                "page": point.payload.get("page", 1),
                "source": point.payload.get("source", "unknown"),
                "type": point.payload.get("type", "text"),
                "user_id": point.payload.get("user_id", "unknown"),
                "score": point.score
            })
        
        logger.info(f"✅ Found {len(formatted_results)} results from Qdrant")
        return formatted_results
        
    except Exception as e:
        logger.error(f"❌ Qdrant search failed: {e}")
        return []

import numpy as np

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def filter_graph_chunks(graph_chunks, query, embed_model, threshold=0.42):
    """Remove graph chunks that are not semantically relevant to the query."""
    
    if not graph_chunks:
        return []

    q_emb = embed_model.encode(f"query: {query}").tolist()
    
    filtered = []
    for chunk in graph_chunks:
        text = chunk.get("text") or chunk.get("page_content")
        if not text:
            continue

        emb = embed_model.encode(f"passage: {text}").tolist()
        score = cosine_sim(q_emb, emb)

        if score >= threshold:
            chunk["semantic_score"] = float(score)
            filtered.append(chunk)

    return filtered
