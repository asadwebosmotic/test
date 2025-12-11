"""
Pipeline Module - Full PDF Processing Orchestrator

This module orchestrates the complete ingestion pipeline:
1. Extract text from PDF
2. Chunk the extracted content
3. Enrich chunks with spaCy NLP (entities, noun phrases)
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


# def convert_pages_to_chunk_format(extraction_result) -> List[Dict]:
#     """
#     Convert PdfExtractionResult pages to the format expected by chunk_pdfplumber_parsed_data.
    
#     Args:
#         extraction_result: PdfExtractionResult object with pages
        
#     Returns:
#         List of page dictionaries in the expected format
#     """
#     formatted_pages = []
    
#     for page in extraction_result.pages:
#         formatted_page = {
#             "text": page.text,
#             "tables": [],  # pdfplumber table extraction happens during parsing
#             "metadata": {
#                 "page_number": page.page_number,
#                 "source": page.filename
#             }
#         }
#         formatted_pages.append(formatted_page)
    
#     return formatted_pages


# def process_pdf(filepath: str, user_id: str = "anonymous") -> Dict:
#     """
#     Full orchestrator for PDF processing pipeline.
    
#     Steps:
#     1. Run extract() from parsing_and_chunking.py
#     2. Convert pages to the expected format
#     3. Run chunk_pdfplumber_parsed_data()
#     4. Enrich chunks with spaCy NLP (entities, noun phrases)
#     5. Store chunks in Qdrant using embed_and_store_pdf()
#     6. Create document + chunk + entity nodes in Memgraph
    
#     Args:
#         filepath (str): Path to the PDF file
#         user_id (str): User identifier (default: "anonymous")
        
#     Returns:
#         dict: {
#             "document_id": "...",
#             "chunks_count": 42,
#             "entities_count": N,
#             ...
#         }
#     """
#     logger.info(f"🚀 Starting PDF processing pipeline for: {filepath}")
    
#     # Validate file exists
#     if not os.path.exists(filepath):
#         raise FileNotFoundError(f"PDF file not found: {filepath}")
    
#     filename = os.path.basename(filepath)
    
#     # Step 1: Extract text from PDF
#     logger.info("📄 Step 1: Extracting text from PDF...")
#     extraction_result = extract(filepath)
    
#     if not extraction_result.pages:
#         raise ValueError("No content could be extracted from the PDF")
    
#     logger.info(f"✅ Extracted {len(extraction_result.pages)} pages")
    
#     # Step 2: Convert pages to chunk format
#     logger.info("🔄 Step 2: Converting pages to chunk format...")
#     formatted_pages = convert_pages_to_chunk_format(extraction_result)
    
#     # Step 3: Chunk the content
#     logger.info("✂️ Step 3: Chunking content...")
#     chunks = chunk_pdfplumber_parsed_data(formatted_pages)
    
#     if not chunks:
#         raise ValueError("No chunks could be created from the PDF content")
    
#     logger.info(f"✅ Created {len(chunks)} chunks")
    
#     # Step 4: Enrich chunks with spaCy NLP
#     logger.info("🧠 Step 4: Enriching chunks with spaCy NLP...")
#     enriched_chunks = enrich_chunks_batch(chunks)
    
#     # Count total entities and phrases
#     total_entities = sum(len(c.get("metadata", {}).get("entities", [])) for c in enriched_chunks)
#     total_phrases = sum(len(c.get("metadata", {}).get("phrases", [])) for c in enriched_chunks)
#     logger.info(f"✅ Extracted {total_entities} entities and {total_phrases} noun phrases")
    
#     # Step 5: Store chunks in Qdrant (vector embeddings)
#     logger.info("📦 Step 5: Storing chunks in Qdrant...")
#     embed_and_store_pdf(enriched_chunks, user_id=user_id)
#     logger.info("✅ Chunks stored in Qdrant")
    
#     # Step 6: Create document and chunk nodes in Memgraph
#     logger.info("🕸️ Step 6: Creating graph nodes in Memgraph...")
#     document_id = store_document_in_memgraph(filename, user_id)
#     stored_chunks_count = store_chunks_batch_in_memgraph(document_id, enriched_chunks)
#     logger.info(f"✅ Graph nodes created in Memgraph")
    
#     result = {
#         "document_id": document_id,
#         "chunks_count": len(enriched_chunks),
#         "filename": filename,
#         "user_id": user_id,
#         "pages_extracted": len(extraction_result.pages),
#         "graph_chunks_stored": stored_chunks_count,
#         "entities_extracted": total_entities,
#         "phrases_extracted": total_phrases
#     }
    
#     logger.info(f"🎉 Pipeline complete! Document ID: {document_id}, "
#                 f"Chunks: {len(enriched_chunks)}, Entities: {total_entities}")
    
#     return result


# async def process_pdf_async(filepath: str, user_id: str = "anonymous") -> Dict:
#     """
#     Async wrapper for PDF processing pipeline.
    
#     This is useful for FastAPI endpoints to avoid blocking.
#     """
#     import asyncio
    
#     # Run the synchronous function in a thread pool
#     loop = asyncio.get_event_loop()
#     result = await loop.run_in_executor(None, process_pdf, filepath, user_id)
    
#     return result

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

    # === 4. spaCy NLP enrichment ===
    enriched_chunks = enrich_chunks_batch(chunks)

    total_entities = sum(len(c.get("metadata", {}).get("entities", [])) 
                         for c in enriched_chunks)

    logger.info(f"🧠 Extracted total {total_entities} entities")

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