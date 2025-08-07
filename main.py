from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Haven Embedding Service",
    description="Microservice for creating embeddings using sentence transformers",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model: Optional[SentenceTransformer] = None


# Pydantic models for request/response
class EmbeddingRequest(BaseModel):
    text: str
    model_name: Optional[str] = "all-MiniLM-L6-v2"


class EmbeddingResponse(BaseModel):
    embedding: List[float]
    model_name: str
    text_length: int


class BatchEmbeddingRequest(BaseModel):
    texts: List[str]
    model_name: Optional[str] = "all-MiniLM-L6-v2"


class BatchEmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    model_name: str
    text_count: int


@app.on_event("startup")
async def startup_event():
    """Initialize the sentence transformer model on startup"""
    global model
    try:
        logger.info("Loading sentence transformer model...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise e


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Haven Embedding Service",
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_name": model.get_sentence_embedding_dimension() if model else None,
    }


@app.post("/embed", response_model=EmbeddingResponse)
async def create_embedding(request: EmbeddingRequest):
    """Create embedding for a single text"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Create embedding
        embedding = model.encode(request.text)

        return EmbeddingResponse(
            embedding=embedding.tolist(),
            model_name=request.model_name,
            text_length=len(request.text),
        )
    except Exception as e:
        logger.error(f"Error creating embedding: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error creating embedding: {str(e)}"
        )


@app.post("/embed/batch", response_model=BatchEmbeddingResponse)
async def create_batch_embeddings(request: BatchEmbeddingRequest):
    """Create embeddings for multiple texts"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Create embeddings for all texts
        embeddings = model.encode(request.texts)

        return BatchEmbeddingResponse(
            embeddings=embeddings.tolist(),
            model_name=request.model_name,
            text_count=len(request.texts),
        )
    except Exception as e:
        logger.error(f"Error creating batch embeddings: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error creating batch embeddings: {str(e)}"
        )


@app.get("/models")
async def list_models():
    """List available models"""
    return {
        "available_models": [
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2",
            "multi-qa-MiniLM-L6-cos-v1",
        ],
        "current_model": "all-MiniLM-L6-v2",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=3016)
