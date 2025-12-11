"""
NLP Engine Module - spaCy NLP Processing

This module handles NLP processing using spaCy for:
- Entity extraction
- Noun phrase extraction
- Query enhancement
"""
import logging
from typing import List, Tuple, Dict, Optional
import spacy
from spacy.tokens import Doc

# === Configure logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Load spaCy model ===
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("✅ Loaded spaCy model: en_core_web_sm")
except OSError:
    logger.warning("⚠️ spaCy model not found, downloading en_core_web_sm...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")
    logger.info("✅ Downloaded and loaded spaCy model: en_core_web_sm")


def extract_entities(chunk_text): #-> List[Tuple[str, str]]:
    """
    Extract named entities from text using spaCy.
    
    Args:
        text (str): Input text
        
    Returns:
        List[Tuple[str, str]]: List of (entity_text, entity_label) tuples
    """
    try:
        doc = nlp(chunk_text)
        entities = [ent.text for ent in doc.ents if ent.label_ not in ["CARDINAL"]]
        return list(set(entities))
        # entities = [(ent.text.strip(), ent.label_) for ent in doc.ents if ent.text.strip()]
        # return entities
    except Exception as e:
        logger.warning(f"⚠️ Entity extraction failed: {e}")
        return []


def extract_noun_phrases(text: str) -> List[str]:
    """
    Extract noun phrases from text using spaCy.
    
    Args:
        text (str): Input text
        
    Returns:
        List[str]: List of noun phrase texts
    """
    try:
        doc = nlp(text)
        phrases = [chunk.text.strip() for chunk in doc.noun_chunks if chunk.text.strip()]
        return phrases
    except Exception as e:
        logger.warning(f"⚠️ Noun phrase extraction failed: {e}")
        return []


def process_text_with_spacy(text: str) -> Dict:
    """
    Full NLP processing of text with spaCy.
    
    Args:
        text (str): Input text
        
    Returns:
        dict: {
            "entities": [(text, label), ...],
            "phrases": [phrase1, phrase2, ...]
        }
    """
    entities = extract_entities(text)
    phrases = extract_noun_phrases(text)
    
    return {
        "entities": entities,
        "phrases": phrases
    }


def enhance_query_with_nlp(query: str) -> Dict:
    """
    Process a query with spaCy and prepare enhanced query components.
    
    Args:
        query (str): The user query
        
    Returns:
        dict: {
            "original_query": str,
            "entities": [(text, label), ...],
            "entity_texts": [text1, text2, ...],
            "phrases": [phrase1, phrase2, ...],
            "enhanced_query": str  # Original + entities + phrases
        }
    """
    nlp_result = process_text_with_spacy(query)
    
    entity_texts = [ent[0] for ent in nlp_result["entities"]]
    phrases = nlp_result["phrases"]
    
    # Build enhanced query string
    enhanced_parts = [query]
    enhanced_parts.extend(entity_texts)
    enhanced_parts.extend(phrases)
    
    enhanced_query = " ".join(enhanced_parts)
    
    logger.info(f"🧠 NLP processed query: entities={entity_texts}, phrases={phrases}")
    
    return {
        "original_query": query,
        "entities": nlp_result["entities"],
        "entity_texts": entity_texts,
        "phrases": phrases,
        "enhanced_query": enhanced_query
    }


def enrich_chunk_with_nlp(chunk: Dict) -> Dict:
    """
    Enrich a chunk with spaCy-extracted entities and noun phrases.
    
    Args:
        chunk (dict): Chunk with page_content and metadata
        
    Returns:
        dict: Same chunk with added entities and phrases in metadata
    """
    text = chunk.get("page_content", "")
    
    if not text:
        return chunk
    
    nlp_result = process_text_with_spacy(text)
    
    # Add to metadata
    if "metadata" not in chunk:
        chunk["metadata"] = {}
    
    chunk["metadata"]["entities"] = nlp_result["entities"]
    chunk["metadata"]["phrases"] = nlp_result["phrases"]
    
    return chunk


def enrich_chunks_batch(chunks: List[Dict]) -> List[Dict]:
    """
    Enrich multiple chunks with spaCy NLP in batch.
    
    Args:
        chunks (List[Dict]): List of chunks
        
    Returns:
        List[Dict]: Enriched chunks
    """
    logger.info(f"🧠 Enriching {len(chunks)} chunks with spaCy NLP...")
    
    enriched_chunks = []
    for chunk in chunks:
        enriched_chunk = enrich_chunk_with_nlp(chunk)
        enriched_chunks.append(enriched_chunk)
    
    # Log summary
    total_entities = sum(len(c.get("metadata", {}).get("entities", [])) for c in enriched_chunks)
    total_phrases = sum(len(c.get("metadata", {}).get("phrases", [])) for c in enriched_chunks)
    
    logger.info(f"✅ Extracted {total_entities} entities and {total_phrases} phrases from {len(chunks)} chunks")
    
    return enriched_chunks
