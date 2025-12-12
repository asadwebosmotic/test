"""
Pipeline Module - Full PDF Processing Orchestrator

This module orchestrates the complete ingestion pipeline:
1. Extract text from PDF
2. Chunk the extracted content
3. Enrich chunks with Gemini LLM (entities, noun phrases)
4. Store chunks in Qdrant (vector embeddings)
5. Store document, chunks, and entities in Memgraph (graph)
"""
import os
import logging
from typing import Dict, List

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from RAG.parsing_and_chunking import extract, chunk_pdfplumber_parsed_data
from RAG.embedding_and_store import embed_and_store_pdf
from RAG.graph_store import store_document_in_memgraph, store_chunks_batch_in_memgraph
from RAG.nlp_engine import enrich_chunks_batch

# === Configure logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_pages_to_chunk_format(extraction_result) -> List[Dict]:
    formatted_pages = []

    for page in extraction_result.pages:
        formatted_pages.append({
            "text": page.text,
            "tables": [],
            "metadata": {
                "page_number": page.page_number,
                "source": page.filename
            }
        })

    return formatted_pages



def process_pdf(filepath: str, user_id: str = "anonymous") -> Dict:
    logger.info(f"🚀 Starting PDF processing for: {filepath}")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"PDF not found: {filepath}")

    filename = os.path.basename(filepath)

    # === 1. Extract ===
    extraction_result = extract(filepath)
    logger.info(f"📄 Extracted {len(extraction_result.pages)} pages")

    # === 2. Format pages for chunking ===
    formatted_pages = convert_pages_to_chunk_format(extraction_result)

    # === 3. Chunk ===
    chunks = chunk_pdfplumber_parsed_data(formatted_pages)
    logger.info(f"✂️ Created {len(chunks)} chunks")

    # === 4. Gemini LLM entity enrichment ===
    enriched_chunks = enrich_chunks_batch(chunks)

    # === Log extracted entities per chunk ===
    total_entities = 0
    total_phrases = 0
    for i, chunk in enumerate(enriched_chunks):
        entities = chunk.get("metadata", {}).get("entities", [])
        phrases = chunk.get("metadata", {}).get("phrases", [])
        total_entities += len(entities)
        total_phrases += len(phrases)
        
        # Log each chunk's extractions
        page = chunk.get("metadata", {}).get("page_number", "?")
        chunk_preview = chunk.get("page_content", "")[:50]
        logger.info(f"📋 Chunk {i+1} (page {page}): '{chunk_preview}...'")
        logger.info(f"   ├── Entities ({len(entities)}): {entities[:10]}{'...' if len(entities) > 10 else ''}")
        logger.info(f"   └── Phrases ({len(phrases)}): {phrases[:5]}{'...' if len(phrases) > 5 else ''}")

    logger.info(f"🧠 Extracted total {total_entities} entities and {total_phrases} phrases from {len(enriched_chunks)} chunks")

    # === 5. Store in Qdrant ===
    embed_and_store_pdf(enriched_chunks, user_id=user_id)
    logger.info("📦 Chunks stored in Qdrant")

    # === 6. Store in Memgraph ===
    document_id = store_document_in_memgraph(filename, user_id)
    stored_chunks = store_chunks_batch_in_memgraph(document_id, enriched_chunks)

    logger.info("🕸️ Graph stored in Memgraph successfully")

    return {
        "document_id": document_id,
        "chunks_count": len(enriched_chunks),
        "pages_extracted": len(extraction_result.pages),
        "user_id": user_id,
        "entities_extracted": total_entities,
        "graph_chunks_stored": stored_chunks,
        "filename": filename,
    }



async def process_pdf_async(filepath: str, user_id: str = "anonymous") -> Dict:
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, process_pdf, filepath, user_id)