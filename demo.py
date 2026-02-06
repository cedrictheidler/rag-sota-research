#!/usr/bin/env python3
"""
LightRAG Demo - Using Ollama for local inference
"""

import os
import asyncio
from functools import partial
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc

# Configuration
WORKING_DIR = "./rag_data"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
EMBED_DIM = 768

os.makedirs(WORKING_DIR, exist_ok=True)


async def main():
    print("LightRAG Demo")
    print(f"Ollama: {OLLAMA_HOST}")
    print(f"LLM: {LLM_MODEL}")
    print(f"Embeddings: {EMBED_MODEL}")

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,
        llm_model_name=LLM_MODEL,
        llm_model_kwargs={"host": OLLAMA_HOST, "timeout": 300},
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBED_DIM,
            max_token_size=8192,
            func=partial(
                ollama_embed.func,
                embed_model=EMBED_MODEL,
                host=OLLAMA_HOST,
            ),
        ),
    )
    
    await rag.initialize_storages()

    # Sample document
    sample_doc = """
    # Example Document
    Your content here...
    """

    print("Inserting document...")
    await rag.ainsert(sample_doc)
    print("Done!")

    result = await rag.aquery(
        "What is this document about?",
        param=QueryParam(mode="hybrid")
    )
    print(f"Answer: {result}")


if __name__ == "__main__":
    asyncio.run(main())
