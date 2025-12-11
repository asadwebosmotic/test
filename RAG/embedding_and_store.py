import logging
import os
import uuid
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

# === Configure logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Initialize Qdrant client and models ===
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
embed_model = SentenceTransformer("intfloat/e5-base-v2")

# === Recreate collection if not exists ===
if not client.collection_exists("KnowMe_chunks"):
    client.create_collection(
        collection_name="KnowMe_chunks",
        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
    )

# === Core Function to Embed and Store PDF Data ===
def embed_and_store_pdf(chunks: List[dict], user_id: str = "anonymous") -> List[PointStruct]:
    """
    embeds and stores the given PDF into Qdrant.
    
    Args:
        chunks (List[dict]): List of text chunks to embed and store.
        user_id (str): User identifier for data isolation.

    Returns:
        List[PointStruct]: Points that were embedded and stored.
    """

    # === Create Embeddings ===
    data = [chunk.get("page_content", "") for chunk in chunks]
    embeddings = embed_model.encode(data).tolist()

    # === Build + Upsert Points to Qdrant ===
    points = []
    for emb, chunk, text in zip(embeddings, chunks, data):
        if text.strip():
            metadata = chunk.get("metadata", {})
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=emb,
                payload={
                    "text": text,
                    "page": metadata.get("page_number", 1),
                    "source": os.path.basename(metadata.get("source", "unknown")),
                    "type": metadata.get("type", "text"),
                    "user_id": user_id  # Add user ID for data isolation
                }
            )
            points.append(point)
            logger.info(f"Embedded chunk: {text[:100]}... with metadata: {point.payload}")

    client.upsert(collection_name='KnowMe_chunks', points=points)
    logger.info(f"📦 Stored {len(points)} chunks into Qdrant for user {user_id}.")

    return points  # Useful for testing or future chaining (e.g. rerank preview