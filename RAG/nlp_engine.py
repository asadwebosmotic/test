"""
NLP Engine Module - Gemini LLM-based NLP Processing

This module handles NLP processing using Google Gemini 2.5 Flash LLM for:
- Entity extraction from document chunks
- Noun phrase/key concept extraction
- Query enhancement

Replaces the previous spaCy-based implementation with LLM-based extraction
for improved entity recognition quality.
"""
import logging
import json
import re
from typing import List, Tuple, Dict, Optional
import google.generativeai as genai

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

# === Configure logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Configure Gemini ===
genai.configure(api_key=settings.GOOGLE_API_KEY)

# Prompt template for entity extraction (for document chunks)
ENTITY_EXTRACTION_PROMPT = """
You are an advanced information extraction system designed to analyze ANY type of document
(PDFs, reports, CVs, invoices, medical scans, legal papers, research papers, etc.)
and extract high-quality structured knowledge.

Your task is to extract:
1. **Named Entities** → specific, meaningful nouns or noun phrases.
2. **Key Concepts / Key Phrases** → domain-relevant terms that help retrieval or graph construction.

Extract entities intelligently depending on the document type.

===========================
WHAT TO EXTRACT (Entities)
===========================
Extract ONLY meaningful, useful, retrieval-friendly entities such as:

• People names  
• Organizations, hospitals, companies  
• Locations, countries, cities, addresses  
• Dates, timelines, years  
• Measurements (sizes, values, lab measurements, physical numbers)  
• Medical findings, diseases, anatomy terms  
• Legal terms, contract sections  
• Technical concepts, algorithms, frameworks  
• Product names, materials, components  
• Financial terms, invoice items, monetary values  
• Projects, tools, technologies  
• Academic terms, research concepts  
• Events, actions, responsibilities  
• Section-type entities like:
  "Contact Information", "Work Experience", "Diagnosis", "Findings", 
  "Impression", "Summary", "Recommendations"

===========================
WHAT NOT TO EXTRACT
===========================
Do NOT return:
• Stopwords
• Single letters
• Purely common nouns like “document”, “page”, “information”
• Random adjectives or filler text
• Unrelated noise phrases
• Generic verbs like "is", "was", "has"
• Repetitive or overly broad terms

===========================
KEY PHRASES EXTRACTION
===========================
Extract useful domain-relevant phrases:
• Medical: “mesenteric panniculitis”, “urinary bladder wall thickening”
• Legal: “termination clause”, “liability limitation”
• Technical: “vector search pipeline”, “agentic workflow”
• Business: “payment terms”, “project deliverables”
• Academic: “machine learning model performance”

===========================
OUTPUT FORMAT
===========================
Return ONLY a valid JSON object, no explanations, no markdown.

{{
  "entities": ["Entity1", "Entity2", ...],
  "phrases": ["Key phrase 1", "Key phrase 2", ...]
}}

If you are unsure, include more entities rather than fewer — but keep them meaningful.

Text to analyze:
{text}
"""

# Prompt template for query enhancement
QUERY_ENHANCEMENT_PROMPT = """You are an expert at understanding user queries for document retrieval. Analyze the query and extract search terms.

Query:
{query}

Instructions:
1. Extract NAMED ENTITIES mentioned (people, organizations, places, etc.)
2. Identify INFORMATION TYPES being requested:
   - If asking about phone/email/address → include "Contact Information"
   - If asking about jobs/roles/companies → include "Work Experience"
   - If asking about degrees/schools → include "Education"
   - If asking about skills/technologies → include "Skills"
   - If asking about projects → include "Projects"
3. Extract KEY SEARCH TERMS that would help find relevant documents

Return your response as a JSON object with this exact format:
{{
    "entities": ["Person Name", "Organization", ...],
    "phrases": ["Contact Information", "key search term", ...]
}}

IMPORTANT: Return ONLY the JSON object, no additional text or explanation."""


def _parse_llm_json_response(response_text: str) -> Dict:
    """
    Parse JSON response from LLM, handling potential formatting issues.
    Handles markdown code blocks, extra whitespace, and various formats.
    
    Args:
        response_text (str): Raw LLM response
        
    Returns:
        dict: Parsed JSON with 'entities' and 'phrases' keys
    """
    if not response_text:
        return {"entities": [], "phrases": []}
    
    text = response_text.strip()
    
    # Strategy 1: Try direct JSON parsing first
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Extract from markdown code blocks (```json ... ``` or ``` ... ```)
    # Use a more robust pattern that handles newlines properly
    code_block_patterns = [
        r'```json\s*\n?([\s\S]*?)\n?\s*```',  # ```json ... ```
        r'```\s*\n?([\s\S]*?)\n?\s*```',       # ``` ... ```
    ]
    
    for pattern in code_block_patterns:
        try:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                json_content = match.group(1).strip()
                result = json.loads(json_content)
                if isinstance(result, dict):
                    logger.debug(f"✅ Parsed JSON from markdown code block")
                    return result
        except json.JSONDecodeError as e:
            logger.debug(f"Code block parse failed: {e}")
            continue
    
    # Strategy 3: Find JSON object by matching braces
    try:
        # Find the first { and last }
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = text[start_idx:end_idx + 1]
            result = json.loads(json_str)
            if isinstance(result, dict):
                logger.debug(f"✅ Parsed JSON by brace matching")
                return result
    except json.JSONDecodeError as e:
        logger.debug(f"Brace matching parse failed: {e}")
    
    # Strategy 4: Try to manually build arrays from common patterns
    # This handles cases where JSON is malformed but content is extractable
    try:
        entities = []
        phrases = []
        
        # Look for "entities": [...] pattern
        entities_match = re.search(r'"entities"\s*:\s*\[([\s\S]*?)\]', text)
        if entities_match:
            entities_str = entities_match.group(1)
            # Extract quoted strings
            entities = re.findall(r'"([^"]+)"', entities_str)
        
        # Look for "phrases": [...] pattern
        phrases_match = re.search(r'"phrases"\s*:\s*\[([\s\S]*?)\]', text)
        if phrases_match:
            phrases_str = phrases_match.group(1)
            phrases = re.findall(r'"([^"]+)"', phrases_str)
        
        if entities or phrases:
            logger.debug(f"✅ Extracted via regex fallback: {len(entities)} entities, {len(phrases)} phrases")
            return {"entities": entities, "phrases": phrases}
            
    except Exception as e:
        logger.debug(f"Regex fallback failed: {e}")
    
    # If all parsing fails, log and return empty
    logger.warning(f"⚠️ Failed to parse LLM JSON response: {text[:200]}...")
    return {"entities": [], "phrases": []}


def extract_entities(chunk_text: str) -> List[str]:
    """
    Extract named entities from text using Gemini LLM.
    
    Args:
        chunk_text (str): Input text
        
    Returns:
        List[str]: List of entity texts
    """
    if not chunk_text or len(chunk_text.strip()) < 10:
        return []
    
    try:
        model = genai.GenerativeModel(settings.GEMINI_MODEL)
        
        prompt = ENTITY_EXTRACTION_PROMPT.format(text=chunk_text[:4000])  # Limit text length
        
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.1,  # Low temperature for consistent extraction
                max_output_tokens=1024
            )
        )
        
        result = _parse_llm_json_response(response.text)
        entities = result.get("entities", [])
        
        # Deduplicate and clean entities
        entities = list(set([e.strip() for e in entities if e and len(e.strip()) > 1]))
        
        return entities
        
    except Exception as e:
        logger.warning(f"⚠️ LLM entity extraction failed: {e}")
        return []


def extract_noun_phrases(text: str) -> List[str]:
    """
    Extract key phrases/concepts from text using Gemini LLM.
    
    Args:
        text (str): Input text
        
    Returns:
        List[str]: List of key phrase texts
    """
    if not text or len(text.strip()) < 10:
        return []
    
    try:
        model = genai.GenerativeModel(settings.GEMINI_MODEL)
        
        prompt = ENTITY_EXTRACTION_PROMPT.format(text=text[:4000])
        
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                max_output_tokens=1024
            )
        )
        
        result = _parse_llm_json_response(response.text)
        phrases = result.get("phrases", [])
        
        # Deduplicate and clean phrases
        phrases = list(set([p.strip() for p in phrases if p and len(p.strip()) > 1]))
        
        return phrases
        
    except Exception as e:
        logger.warning(f"⚠️ LLM phrase extraction failed: {e}")
        return []


def process_text_with_llm(text: str) -> Dict:
    """
    Full NLP processing of text with Gemini LLM.
    Makes a single LLM call to extract both entities and phrases.
    
    Args:
        text (str): Input text
        
    Returns:
        dict: {
            "entities": [entity1, entity2, ...],
            "phrases": [phrase1, phrase2, ...]
        }
    """
    if not text or len(text.strip()) < 10:
        return {"entities": [], "phrases": []}
    
    try:
        model = genai.GenerativeModel(settings.GEMINI_MODEL)
        
        prompt = ENTITY_EXTRACTION_PROMPT.format(text=text[:4000])
        
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                max_output_tokens=1024
            )
        )
        
        result = _parse_llm_json_response(response.text)
        
        entities = list(set([e.strip() for e in result.get("entities", []) if e and len(e.strip()) > 1]))
        phrases = list(set([p.strip() for p in result.get("phrases", []) if p and len(p.strip()) > 1]))
        
        return {
            "entities": entities,
            "phrases": phrases
        }
        
    except Exception as e:
        logger.warning(f"⚠️ LLM text processing failed: {e}")
        return {"entities": [], "phrases": []}


def enhance_query_with_nlp(query: str) -> Dict:
    """
    Process a query with Gemini LLM and prepare enhanced query components.
    
    Args:
        query (str): The user query
        
    Returns:
        dict: {
            "original_query": str,
            "entities": [entity1, entity2, ...],
            "entity_texts": [entity1, entity2, ...],
            "phrases": [phrase1, phrase2, ...],
            "enhanced_query": str  # Original + entities + phrases
        }
    """
    if not query or len(query.strip()) < 3:
        return {
            "original_query": query,
            "entities": [],
            "entity_texts": [],
            "phrases": [],
            "enhanced_query": query
        }
    
    try:
        model = genai.GenerativeModel(settings.GEMINI_MODEL)
        
        prompt = QUERY_ENHANCEMENT_PROMPT.format(query=query)
        
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                max_output_tokens=512
            )
        )
        
        result = _parse_llm_json_response(response.text)
        
        entities = list(set([e.strip() for e in result.get("entities", []) if e and len(e.strip()) > 1]))
        phrases = list(set([p.strip() for p in result.get("phrases", []) if p and len(p.strip()) > 1]))
        
        # Build enhanced query string
        enhanced_parts = [query]
        enhanced_parts.extend(entities)
        enhanced_parts.extend(phrases)
        
        enhanced_query = " ".join(enhanced_parts)
        
        logger.info(f"🧠 LLM processed query: entities={entities}, phrases={phrases}")
        
        return {
            "original_query": query,
            "entities": entities,
            "entity_texts": entities,  # Same as entities for compatibility
            "phrases": phrases,
            "enhanced_query": enhanced_query
        }
        
    except Exception as e:
        logger.warning(f"⚠️ LLM query enhancement failed: {e}")
        return {
            "original_query": query,
            "entities": [],
            "entity_texts": [],
            "phrases": [],
            "enhanced_query": query
        }


def enrich_chunk_with_nlp(chunk: Dict) -> Dict:
    """
    Enrich a chunk with LLM-extracted entities and key phrases.
    
    Args:
        chunk (dict): Chunk with page_content and metadata
        
    Returns:
        dict: Same chunk with added entities and phrases in metadata
    """
    text = chunk.get("page_content", "")
    
    if not text:
        return chunk
    
    nlp_result = process_text_with_llm(text)
    
    # Add to metadata
    if "metadata" not in chunk:
        chunk["metadata"] = {}
    
    chunk["metadata"]["entities"] = nlp_result["entities"]
    chunk["metadata"]["phrases"] = nlp_result["phrases"]
    
    return chunk


def enrich_chunks_batch(chunks: List[Dict]) -> List[Dict]:
    """
    Enrich multiple chunks with Gemini LLM in batch.
    
    Uses individual LLM calls for each chunk (Gemini API handles rate limiting).
    For large batches, consider implementing batching or async processing.
    
    Args:
        chunks (List[Dict]): List of chunks
        
    Returns:
        List[Dict]: Enriched chunks
    """
    logger.info(f"🧠 Enriching {len(chunks)} chunks with Gemini LLM...")
    
    enriched_chunks = []
    for i, chunk in enumerate(chunks):
        try:
            enriched_chunk = enrich_chunk_with_nlp(chunk)
            enriched_chunks.append(enriched_chunk)
            
            # Log progress for large batches
            if (i + 1) % 10 == 0:
                logger.info(f"   Processed {i + 1}/{len(chunks)} chunks...")
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to enrich chunk {i}: {e}")
            # Add chunk without enrichment on failure
            chunk.setdefault("metadata", {})
            chunk["metadata"]["entities"] = []
            chunk["metadata"]["phrases"] = []
            enriched_chunks.append(chunk)
    
    # Log summary
    total_entities = sum(len(c.get("metadata", {}).get("entities", [])) for c in enriched_chunks)
    total_phrases = sum(len(c.get("metadata", {}).get("phrases", [])) for c in enriched_chunks)
    
    logger.info(f"✅ Extracted {total_entities} entities and {total_phrases} phrases from {len(chunks)} chunks using Gemini LLM")
    
    return enriched_chunks


# === Backward compatibility aliases ===
# These ensure existing code that references "spacy" functions continue to work
process_text_with_spacy = process_text_with_llm
