"""
Main FastAPI Application - Hybrid RAG Backend Service

This module provides REST API endpoints for:
- PDF upload and processing
- Question answering using Hybrid Retrieval (Vector RAG + GraphRAG)
- Enhanced with spaCy NLP for entity/phrase extraction
"""
import os
import logging
import tempfile
import shutil
from typing import Optional, List
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from RAG.pipeline import process_pdf_async
from RAG.vector_search import search_qdrant, search_qdrant_enhanced
from RAG.graph_search import expand_multi_chunk_context, get_chunks_by_entities
from RAG.llm_engine import answer_with_gemini_async
from RAG.nlp_engine import enhance_query_with_nlp

# === Configure logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Initialize FastAPI app ===
app = FastAPI(
    title="MemeGraph - Hybrid RAG Backend",
    description="""
    A backend service for PDF upload and question answering using 
    Hybrid Retrieval combining Vector RAG (Qdrant) and GraphRAG (Memgraph) 
    with Gemini 2.5 Flash for answer generation.
    
    Enhanced with spaCy NLP for:
    - Entity extraction during document processing
    - Query enhancement using entities and noun phrases
    - Entity-based graph retrieval
    """,
    version="2.0.0"
)

# === CORS Middleware ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === Response Models ===
class UploadResponse(BaseModel):
    """Response model for PDF upload endpoint"""
    success: bool
    document_id: str
    filename: str
    chunks_count: int
    pages_extracted: int
    entities_extracted: int
    phrases_extracted: int
    message: str


class AskResponse(BaseModel):
    """Response model for question answering endpoint"""
    answer: str
    sources: List[dict]
    query: str
    user_id: str
    nlp_entities: List[str]
    nlp_phrases: List[str]


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    message: str


# === Endpoints ===

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="MemeGraph Hybrid RAG Backend is running"
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="All systems operational"
    )


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    user_id: str = Query(default="anonymous", description="User identifier for data isolation")
):
    """
    Upload a PDF file for processing.
    
    This endpoint:
    1. Saves the uploaded PDF temporarily
    2. Extracts text and chunks content
    3. Enriches chunks with spaCy NLP (entities, noun phrases)
    4. Stores chunks in Qdrant (vector embeddings)
    5. Creates document, chunk, and entity nodes in Memgraph (graph)
    
    Args:
        file: The PDF file to upload
        user_id: User identifier (default: "anonymous")
        
    Returns:
        UploadResponse with document summary
    """
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted"
        )
    
    logger.info(f"📤 Received upload: {file.filename} from user: {user_id}")
    
    # Create temporary file to store the upload
    temp_dir = tempfile.mkdtemp()
    temp_filepath = os.path.join(temp_dir, file.filename)
    
    try:
        # Save uploaded file
        with open(temp_filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"💾 Saved temporary file: {temp_filepath}")
        
        # Process the PDF
        result = await process_pdf_async(temp_filepath, user_id=user_id)
        
        return UploadResponse(
            success=True,
            document_id=result["document_id"],
            filename=result["filename"],
            chunks_count=result["chunks_count"],
            pages_extracted=result["pages_extracted"],
            entities_extracted=result.get("entities_extracted", 0),
            phrases_extracted=result.get("phrases_extracted", 0),
            message=f"Successfully processed {result['filename']}: "
                    f"{result['pages_extracted']} pages, {result['chunks_count']} chunks, "
                    f"{result.get('entities_extracted', 0)} entities"
        )
        
    except Exception as e:
        logger.error(f"❌ Upload processing failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process PDF: {str(e)}"
        )
        
    finally:
        # Cleanup temporary files
        try:
            shutil.rmtree(temp_dir)
            logger.info("🧹 Cleaned up temporary files")
        except Exception as e:
            logger.warning(f"⚠️ Failed to cleanup temp files: {e}")


@app.get("/ask", response_model=AskResponse)
async def ask_question(
    query: str = Query(..., description="The question to ask"),
    user_id: str = Query(default="anonymous", description="User identifier for filtering documents")
):
    """
    Ask a question and get an answer using Hybrid Retrieval with spaCy enhancement.
    
    This endpoint:
    1. Parse query with spaCy to extract entities and noun phrases
    2. Enhanced vector search via Qdrant using enriched query
    3. Entity-based graph retrieval from Memgraph
    4. Document-level graph expansion for related chunks
    5. Merge + dedupe all contexts → pass to Gemini
    6. Return answer with sources
    
    Args:
        query: The question to ask
        user_id: User identifier for filtering (default: "anonymous")
        
    Returns:
        AskResponse with answer, sources, and NLP metadata
    """
    logger.info(f"❓ Received question from user {user_id}: '{query[:50]}...'")
    
    try:
        # Step 1: Parse query with spaCy NLP
        logger.info("🧠 Step 1: Processing query with spaCy NLP...")
        nlp_result = enhance_query_with_nlp(query)
        entity_texts = nlp_result["entity_texts"]
        phrases = nlp_result["phrases"]
        enhanced_query = nlp_result["enhanced_query"]
        
        logger.info(f"📝 Extracted entities: {entity_texts}, phrases: {phrases}")
        
        # Step 2: Enhanced vector search via Qdrant
        logger.info("🔍 Step 2: Enhanced vector search in Qdrant...")
        vector_results = search_qdrant_enhanced(
            query=query,
            user_id=user_id,
            top_k=5,
            entities=entity_texts,
            phrases=phrases
        )
        
        # Early exit if no relevant results - skip graph retrieval for irrelevant queries
        if not vector_results:
            logger.info("⚠️ No relevant documents found - skipping graph retrieval")
            logger.info("🤖 Step 6: Generating answer with Gemini (no context)...")
            answer = await answer_with_gemini_async(query, [])
            
            return AskResponse(
                answer=answer,
                sources=[],
                query=query,
                user_id=user_id,
                nlp_entities=entity_texts,
                nlp_phrases=phrases
            )
        
        # Step 3: Entity-based graph retrieval from Memgraph (only if we have relevant vectors)
        logger.info("🔗 Step 3: Entity-based graph retrieval...")
        entity_chunks = []
        if entity_texts:
            entity_chunks = get_chunks_by_entities(entity_texts, limit_per_entity=3)
            logger.info(f"Found {len(entity_chunks)} chunks via entity graph search")
        
        # Step 4: Document-level graph expansion (only if we have relevant vectors)
        logger.info("🕸️ Step 4: Graph expansion in Memgraph...")
        vector_texts = [r["text"] for r in vector_results if r.get("text")]
        graph_texts = expand_multi_chunk_context(vector_texts, limit_per_chunk=3)
        
        # Step 5: Merge and dedupe all contexts
        logger.info("🔗 Step 5: Merging and deduplicating context...")
        all_contexts = set()
        sources = []
        
        # Add vector search results (highest priority)
        for result in vector_results:
            text = result.get("text", "")
            if text and text not in all_contexts:
                all_contexts.add(text)
                sources.append({
                    "text": text[:200] + "..." if len(text) > 200 else text,
                    "page": result.get("page", 1),
                    "source": result.get("source", "unknown"),
                    "type": result.get("type", "text"),
                    "score": result.get("score", 0),
                    "retrieval_method": "vector_enhanced"
                })
        
        # Add entity-based graph results
        for text in entity_chunks:
            if text and text not in all_contexts:
                all_contexts.add(text)
                sources.append({
                    "text": text[:200] + "..." if len(text) > 200 else text,
                    "retrieval_method": "entity_graph"
                })
        
        # Add document-level graph expansion results
        for text in graph_texts:
            if text and text not in all_contexts:
                all_contexts.add(text)
                sources.append({
                    "text": text[:200] + "..." if len(text) > 200 else text,
                    "retrieval_method": "document_graph"
                })
        
        context_passages = list(all_contexts)
        logger.info(f"📚 Total context passages: {len(context_passages)} "
                   f"(vector: {len(vector_results)}, entity_graph: {len(entity_chunks)}, "
                   f"doc_graph: {len(graph_texts)})")
        
        # Step 6: Generate answer with Gemini
        logger.info("🤖 Step 6: Generating answer with Gemini...")
        answer = await answer_with_gemini_async(query, context_passages)
        
        logger.info("✅ Question answered successfully")
        
        return AskResponse(
            answer=answer,
            sources=sources,
            query=query,
            user_id=user_id,
            nlp_entities=entity_texts,
            nlp_phrases=phrases
        )
        
    except Exception as e:
        logger.error(f"❌ Question answering failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to answer question: {str(e)}"
        )


# === Run the app ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
