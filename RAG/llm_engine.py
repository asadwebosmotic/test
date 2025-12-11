"""
LLM Engine Module - Gemini Integration

This module handles LLM-based answer generation using Google Gemini 2.5 Flash.
"""
import logging
from typing import List, Optional
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

# Prompt template for hybrid RAG
HYBRID_RAG_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based on provided document context.

User Question:
{query}

Document Context:
{context_passages}

Instructions:
1. If the user's message is a greeting (like "hello", "hi", "hey") or casual chitchat, respond naturally and friendly WITHOUT referencing any documents.
2. If document context is provided AND is relevant to the question, answer based on the documents.
3. If document context is "(No relevant document context available)", use your general knowledge but clearly state that the information is not from the uploaded documents.
4. NEVER make up fake document references or sources.
5. Be concise but comprehensive.
6. If asked about something NOT in the documents, say "I don't have information about this in the uploaded documents" rather than inventing content.

Answer:"""

# Prompt for no-context scenarios (greetings, chitchat)
NO_CONTEXT_PROMPT_TEMPLATE = """You are a helpful assistant. The user has uploaded documents but their current question doesn't require document lookup.

User Message:
{query}

Instructions:
1. Respond naturally and helpfully.
2. If it's a greeting, greet them back warmly.
3. If they're asking a general question, answer from your general knowledge.
4. Let them know you can help them with questions about their uploaded documents.

Answer:"""


def answer_with_gemini(
    query: str, 
    context_passages: List[str],
    model_name: Optional[str] = None
) -> str:
    """
    Generate an answer using Google Gemini with the provided context.
    Uses different prompts based on whether relevant context is available.
    
    Args:
        query (str): The user's question
        context_passages (List[str]): List of relevant text passages for context
        model_name (Optional[str]): Model to use (defaults to settings.GEMINI_MODEL)
        
    Returns:
        str: The generated answer
    """
    logger.info(f"🤖 Generating answer with Gemini for query: '{query[:50]}...'")
    
    # Use configured model or provided override
    model_to_use = model_name or settings.GEMINI_MODEL
    
    # Choose prompt based on context availability
    if context_passages and len(context_passages) > 0:
        # We have relevant document context
        formatted_context = "\n\n---\n\n".join([
            f"[Passage {i+1}]:\n{passage}" 
            for i, passage in enumerate(context_passages)
        ])
        prompt = HYBRID_RAG_PROMPT_TEMPLATE.format(
            query=query,
            context_passages=formatted_context
        )
        logger.info(f"📄 Using document context ({len(context_passages)} passages)")
    else:
        # No relevant context - use the no-context prompt
        prompt = NO_CONTEXT_PROMPT_TEMPLATE.format(query=query)
        logger.info("💬 No relevant context - using conversational response")
    
    try:
        # Initialize the model
        model = genai.GenerativeModel(model_to_use)
        
        # Generate response
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.3,  # Lower temperature for more factual responses
                max_output_tokens=2048
            )
        )
        
        answer = response.text.strip()
        logger.info(f"✅ Generated answer with {len(answer)} characters")
        return answer
        
    except Exception as e:
        logger.error(f"❌ Gemini generation failed: {e}")
        return f"I apologize, but I encountered an error while generating the answer: {str(e)}"


async def answer_with_gemini_async(
    query: str, 
    context_passages: List[str],
    model_name: Optional[str] = None
) -> str:
    """
    Async wrapper for Gemini answer generation.
    
    This is useful for FastAPI endpoints to avoid blocking.
    """
    import asyncio
    
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, 
        answer_with_gemini, 
        query, 
        context_passages, 
        model_name
    )
    
    return result


def answer_with_gemini_streaming(
    query: str, 
    context_passages: List[str],
    model_name: Optional[str] = None
):
    """
    Generate answer with streaming response.
    
    Yields:
        str: Chunks of the generated answer
    """
    logger.info(f"🤖 Streaming answer with Gemini for query: '{query[:50]}...'")
    
    model_to_use = model_name or settings.GEMINI_MODEL
    
    # Format context passages
    if context_passages:
        formatted_context = "\n\n---\n\n".join([
            f"[Passage {i+1}]:\n{passage}" 
            for i, passage in enumerate(context_passages)
        ])
    else:
        formatted_context = "(No document context available)"
    
    prompt = HYBRID_RAG_PROMPT_TEMPLATE.format(
        query=query,
        context_passages=formatted_context
    )
    
    try:
        model = genai.GenerativeModel(model_to_use)
        
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.3,
                max_output_tokens=2048
            ),
            stream=True
        )
        
        for chunk in response:
            if chunk.text:
                yield chunk.text
                
    except Exception as e:
        logger.error(f"❌ Gemini streaming failed: {e}")
        yield f"I apologize, but I encountered an error: {str(e)}"
