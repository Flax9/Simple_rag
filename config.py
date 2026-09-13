import os

# Remote Inference Endpoint (replace with your active Kaggle ngrok tunnel URL)
REMOTE_LLM_URL = "https://vibes-maturely-snowcap.ngrok-free.dev/v1"
LLM_MODEL_NAME = "qwen3.5:9b"

# Local Embedding & Vector Store Configurations
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "qwen_knowledge_base"

# Retrieval Settings
TOP_K_RESULTS = 2

