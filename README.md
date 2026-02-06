# State-of-the-Art RAG Research (February 2026)

## Executive Summary

RAG in 2026 has evolved significantly beyond simple "embed → retrieve → generate" pipelines. The frontier has moved to:

1. **Graph-based RAG** (GraphRAG, LightRAG) — Knowledge graphs for complex reasoning
2. **Agentic RAG** — Self-correcting retrieval with multi-step reasoning
3. **Hybrid retrieval** — Combining dense vectors + sparse (BM25) + reranking

---

## The RAG Landscape

### Top 5 Frameworks (2025-2026)

| Rank | Framework | Best For |
|------|-----------|----------|
| 1 | **LangGraph** | Complex agentic orchestration |
| 2 | **Haystack 2.x** | Production/enterprise, auditable |
| 3 | **LlamaIndex** | Data ingestion and indexing |
| 4 | **Pathway** | Real-time streaming data |
| 5 | **Dify** | Low-code rapid development |

### SOTA Techniques

1. **GraphRAG (Microsoft)** — 30.8k stars
   - Builds knowledge graphs from documents
   - Excels at aggregation queries
   - Expensive during indexing

2. **LightRAG (HKU)** — Trending
   - ~10x cheaper than GraphRAG
   - Supports local Ollama models
   - EMNLP 2025 paper

3. **Agentic RAG**
   - Self-correcting retrieval
   - LangGraph is the leading framework

4. **Hybrid Retrieval**
   - Dense + sparse + reranker
   - ColBERT or Cohere Rerank

---

## Quick Start: LightRAG

```bash
uv tool install "lightrag-hku[api]"
```

See `demo.py` for a working example with Ollama.

---

## References

- [Microsoft GraphRAG](https://github.com/microsoft/graphrag)
- [LightRAG (HKU)](https://github.com/HKUDS/LightRAG)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [Haystack](https://haystack.deepset.ai/)
- [LlamaIndex](https://www.llamaindex.ai/)

---

*Research compiled by Cedric — February 6, 2026*
