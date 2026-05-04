"""
local_rag_system.py
────────────────────────────────────────────────────────────────
Drop-in replacement for AdvancedRAGSystem that runs 100% locally.

Replaces:
  NVIDIAEmbedding          → BAAI/bge-large-en-v1.5  (sentence-transformers)
  NVIDIARerank             → BAAI/bge-reranker-large  (FlagEmbedding)
  OpenRouter (DeepSeek)    → Ollama llama3:70b         (or vLLM)
"""

from __future__ import annotations

import os
import asyncio
import tempfile
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import nest_asyncio
import chromadb

# LlamaIndex core
from llama_index.core import (
    SimpleDirectoryReader, VectorStoreIndex,
    StorageContext, Settings, Document
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.workflow import Context
from llama_index.core.schema import NodeWithScore, TextNode

# Local model integrations
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore

# Reranker — prefer SentenceTransformerRerank (works with transformers 5.x + CPU)
# FlagEmbeddingReranker has an incompatibility with transformers>=5 (prepare_for_model removed)
try:
    from llama_index.core.postprocessor import SentenceTransformerRerank
    HAS_ST_RERANKER = True
except ImportError:
    HAS_ST_RERANKER = False
    logging.warning("SentenceTransformerRerank not available; reranker disabled")

nest_asyncio.apply()
logger = logging.getLogger(__name__)


class LocalRAGSystem:
    """
    Fully local RAG system for evaluation.

    Usage
    -----
    rag = LocalRAGSystem(config)
    rag.setup_models()
    rag.ingest_documents_from_texts(texts)   # list[str] or list[Document]
    rag.setup_retrieval()
    results = rag.retrieve(query, return_nodes=True)
    answer  = asyncio.run(rag.query(query))
    """

    def __init__(self, config: dict):
        self.config = config
        self.documents: Optional[List[Document]] = None
        self.nodes: Optional[list] = None
        self.vector_index: Optional[VectorStoreIndex] = None
        self.bm25_retriever: Optional[BM25Retriever] = None
        self._dense_retriever = None       # stored so retrieve() can rebuild fusion weights
        self.hybrid_retriever = None
        self.reranker = None
        self.query_engine: Optional[RetrieverQueryEngine] = None
        self.agent_workflow: Optional[AgentWorkflow] = None
        self.ctx: Optional[Context] = None
        self.is_initialized = False
        self._chroma_client = None
        self._collection_name = "eval_rag_collection"

    # ──────────────────────────────────────────────────────────────────────
    # vLLM init with MockLLM fallback
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _init_vllm_or_mock(llm_cfg: dict):
        """
        Try to connect to a running vLLM OpenAI-compatible server.
        Falls back to MockLLM if the server is not reachable so that
        retrieval metrics still run end-to-end without crashing.

        To get real LLM answers, start the server first:
          python -m vllm.entrypoints.openai.api_server \\
            --model /home/paritosh/casal/models/Meta-Llama-3-8B \\
            --served-model-name meta-llama/Meta-Llama-3-8B \\
            --port 8000 --dtype bfloat16
        """
        import urllib.request
        base_url = llm_cfg.get("base_url", "http://localhost:8000/v1")
        health_url = base_url.rstrip("/v1").rstrip("/") + "/health"
        server_up = False
        try:
            urllib.request.urlopen(health_url, timeout=2)
            server_up = True
        except Exception:
            pass

        if server_up:
            from llama_index.llms.openai_like import OpenAILike
            logger.info(f"vLLM server reachable — connecting to {llm_cfg['model']}")
            return OpenAILike(
                model=llm_cfg["model"],
                api_base=base_url,
                api_key="EMPTY",
                temperature=llm_cfg.get("temperature", 0.0),
                max_tokens=llm_cfg.get("max_tokens", 512),
                is_chat_model=True,
            )

        logger.warning(
            "vLLM server not reachable at %s — using MockLLM.\n"
            "Retrieval metrics will be real; generated_answer will be a stub.\n"
            "Start the server for real answers:\n"
            "  python -m vllm.entrypoints.openai.api_server \\\n"
            "    --model /home/paritosh/casal/models/Meta-Llama-3-8B \\\n"
            "    --served-model-name meta-llama/Meta-Llama-3-8B \\\n"
            "    --port 8000 --dtype bfloat16",
            health_url,
        )
        from llama_index.core.llms import MockLLM
        return MockLLM(max_tokens=llm_cfg.get("max_tokens", 512))

    # ──────────────────────────────────────────────────────────────────────
    # Model initialisation
    # ──────────────────────────────────────────────────────────────────────
    def setup_models(self) -> bool:
        """Initialise all local models."""
        try:
            em_cfg   = self.config["models"]["embedding"]
            llm_cfg  = self.config["models"][self.config["models"]["llm_backend"]]
            rnk_cfg  = self.config["models"]["reranker"]

            # 1. Embedding model
            logger.info(f"Loading embedding model: {em_cfg['model_name']}")
            self.embed_model = HuggingFaceEmbedding(
                model_name=em_cfg["model_name"],
                device=em_cfg.get("device", "cuda"),
                max_length=em_cfg.get("max_length", 512),
                embed_batch_size=em_cfg.get("batch_size", 64),
            )

            # 2. LLM
            backend = self.config["models"]["llm_backend"]
            if backend == "ollama":
                logger.info(f"Connecting to Ollama: {llm_cfg['model']}")
                self.llm = Ollama(
                    model=llm_cfg["model"],
                    base_url=llm_cfg.get("base_url", "http://localhost:11434"),
                    context_window=llm_cfg.get("context_window", 8192),
                    temperature=llm_cfg.get("temperature", 0.0),
                    request_timeout=llm_cfg.get("request_timeout", 120.0),
                )
            elif backend == "vllm":
                self.llm = self._init_vllm_or_mock(llm_cfg)
            else:
                raise ValueError(f"Unknown backend: {backend}")

            # 3. Reranker (SentenceTransformerRerank: explicit device, works with transformers 5.x)
            if HAS_ST_RERANKER:
                logger.info(f"Loading reranker: {rnk_cfg['model_name']}")
                try:
                    self.reranker = SentenceTransformerRerank(
                        model=rnk_cfg["model_name"],
                        top_n=rnk_cfg.get("top_n", 5),
                        device=rnk_cfg.get("device", "cpu"),
                    )
                except Exception as exc:
                    logger.warning(f"Reranker failed to load ({exc}); reranker disabled")
                    self.reranker = None
            else:
                self.reranker = None
                logger.warning("Reranker disabled — llama-index-core postprocessor not found")

            # 4. Global LlamaIndex settings
            Settings.embed_model = self.embed_model
            Settings.llm = self.llm

            logger.info("All local models ready ✓")
            return True

        except Exception as exc:
            logger.error(f"Model setup failed: {exc}", exc_info=True)
            return False

    # ──────────────────────────────────────────────────────────────────────
    # Ingestion
    # ──────────────────────────────────────────────────────────────────────
    def ingest_documents_from_texts(
        self,
        texts: List[str],
        metadata_list: Optional[List[dict]] = None,
        collection_name: Optional[str] = None,
    ) -> Tuple[bool, int, int]:
        """
        Ingest plain text strings as documents.

        Parameters
        ----------
        texts          : list of raw text strings
        metadata_list  : optional per-document metadata dicts
        collection_name: ChromaDB collection name (use unique names per dataset)
        """
        try:
            col_name = collection_name or self._collection_name
            r_cfg = self.config["retrieval"]

            # Build Document objects
            self.documents = [
                Document(
                    text=t,
                    metadata=(metadata_list[i] if metadata_list else {"doc_id": str(i)}),
                )
                for i, t in enumerate(texts)
            ]

            # Chunking
            splitter = SentenceSplitter(
                chunk_size=r_cfg["chunk_size"],
                chunk_overlap=r_cfg["chunk_overlap"],
            )
            self.nodes = splitter.get_nodes_from_documents(self.documents)
            logger.info(f"Chunked {len(self.documents)} docs → {len(self.nodes)} nodes")

            # ChromaDB vector store (in-memory for evaluation speed)
            self._chroma_client = chromadb.EphemeralClient()
            chroma_col = self._chroma_client.get_or_create_collection(col_name)
            vector_store = ChromaVectorStore(chroma_collection=chroma_col)

            # Ingestion pipeline
            pipeline = IngestionPipeline(
                transformations=[
                    SentenceSplitter(
                        chunk_size=r_cfg["chunk_size"],
                        chunk_overlap=r_cfg["chunk_overlap"],
                    ),
                    self.embed_model,
                ],
                vector_store=vector_store,
            )
            pipeline.run(documents=self.documents)

            # Dense index
            self.vector_index = VectorStoreIndex.from_vector_store(
                vector_store, embed_model=self.embed_model
            )

            # BM25 sparse index
            self.bm25_retriever = BM25Retriever.from_defaults(
                nodes=self.nodes,
                similarity_top_k=r_cfg["bm25_top_k"],
            )

            return True, len(self.documents), len(self.nodes)

        except Exception as exc:
            logger.error(f"Ingestion failed: {exc}", exc_info=True)
            return False, 0, 0

    def ingest_documents_from_files(self, file_paths: List[str], **kwargs):
        """Ingest from file paths (PDF / TXT)."""
        reader = SimpleDirectoryReader(input_files=file_paths)
        raw_docs = reader.load_data()
        texts = [d.text for d in raw_docs]
        metas = [d.metadata for d in raw_docs]
        return self.ingest_documents_from_texts(texts, metas, **kwargs)

    # ──────────────────────────────────────────────────────────────────────
    # Retrieval setup
    # ──────────────────────────────────────────────────────────────────────
    def setup_retrieval(self) -> bool:
        """Wire hybrid retriever + reranker + query engine."""
        if not self.vector_index or not self.bm25_retriever:
            logger.error("Call ingest_documents_from_texts first")
            return False
        try:
            r_cfg = self.config["retrieval"]

            self._dense_retriever = self.vector_index.as_retriever(
                similarity_top_k=r_cfg["dense_top_k"]
            )

            self.hybrid_retriever = QueryFusionRetriever(
                [self._dense_retriever, self.bm25_retriever],
                similarity_top_k=r_cfg["hybrid_top_k"],
                num_queries=1,
                retriever_weights=[0.35, 0.65],  # default; overridden dynamically in retrieve()
                mode=r_cfg["fusion_mode"],
                use_async=True,
                verbose=False,
            )

            postprocessors = [self.reranker] if self.reranker else []

            self.query_engine = RetrieverQueryEngine.from_args(
                retriever=self.hybrid_retriever,
                node_postprocessors=postprocessors,
                llm=self.llm,
                response_mode="tree_summarize",
            )
            return True

        except Exception as exc:
            logger.error(f"Retrieval setup failed: {exc}", exc_info=True)
            return False

    # ──────────────────────────────────────────────────────────────────────
    # Query reformulation (legal domain only)
    # ──────────────────────────────────────────────────────────────────────
    def reformulate_query(self, query: str, domain: str = "legal") -> str:
        """
        Rewrite an instruction-style query into keyword-focused search terms.

        CUAD queries read like "Highlight parts related to X that should be
        reviewed by a lawyer" — BGE and BM25 both embed the instruction style
        rather than the legal concept, hurting recall.  This prompt strips
        the instruction wrapper and extracts the core legal keywords.
        Only applied when domain == "legal"; other domains pass through.
        """
        if domain != "legal":
            return query
        prompt = (
            "Rewrite the following as a concise legal search query using specific "
            "keywords only. Remove instruction words like 'highlight', 'find', "
            "'identify'. Keep only the core legal concept and relevant terms.\n"
            f"Original: {query}\n"
            "Search query:"
        )
        try:
            result = self.llm.complete(prompt)
            reformulated = str(result).strip().split("\n")[0].strip()
            # Reject empty, over-long, or MockLLM all-x output
            if (reformulated
                    and 3 < len(reformulated) < 400
                    and reformulated != "x" * len(reformulated)):
                logger.debug("Reformulated: %r → %r", query[:60], reformulated[:60])
                return reformulated
        except Exception as exc:
            logger.warning("Query reformulation failed (%s); using original query", exc)
        return query

    # ──────────────────────────────────────────────────────────────────────
    # Dynamic fusion weight computation
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _compute_weights(query: str, domain: str) -> Tuple[float, float]:
        """
        Compute (dense_weight, bm25_weight) for QueryFusionRetriever.

        Legal contracts have many exact legal terms → favour BM25 (beta=0.65).
        General/medical free-text questions favour semantic dense (alpha=0.75).
        Long queries (>10 words) get an extra BM25 boost (+0.10) because they
        contain more keyword signal.
        """
        if domain == "legal":
            alpha, beta = 0.35, 0.65
        else:
            alpha, beta = 0.75, 0.25

        if len(query.split()) > 10:
            beta = min(beta + 0.10, 0.90)
            alpha = 1.0 - beta

        return alpha, beta

    # ──────────────────────────────────────────────────────────────────────
    # Retrieval (returns nodes for metric computation)
    # ──────────────────────────────────────────────────────────────────────
    def retrieve(self, query: str, domain: str = "general") -> List[NodeWithScore]:
        """
        Run hybrid retrieval + reranking with dynamic fusion weights.
        For legal domain, rewrites instruction-style queries before retrieval.
        Returns a list of NodeWithScore objects.
        """
        if not self._dense_retriever or not self.bm25_retriever:
            raise RuntimeError("Call setup_retrieval() first")

        search_query = self.reformulate_query(query, domain)
        alpha, beta = self._compute_weights(search_query, domain)
        logger.debug("Fusion weights — dense=%.2f  bm25=%.2f  domain=%s  q_len=%d",
                     alpha, beta, domain, len(search_query.split()))

        r_cfg = self.config["retrieval"]
        retriever = QueryFusionRetriever(
            [self._dense_retriever, self.bm25_retriever],
            similarity_top_k=r_cfg["hybrid_top_k"],
            num_queries=1,
            retriever_weights=[alpha, beta],
            mode=r_cfg["fusion_mode"],
            use_async=True,
            verbose=False,
        )

        loop = asyncio.get_event_loop()
        nodes = loop.run_until_complete(retriever.aretrieve(search_query))
        if self.reranker:
            from llama_index.core.schema import QueryBundle
            nodes = self.reranker.postprocess_nodes(
                nodes, query_bundle=QueryBundle(search_query)
            )
        return nodes

    def retrieve_texts(self, query: str, domain: str = "general") -> List[str]:
        """Convenience wrapper — returns list of context strings."""
        return [n.node.get_content() for n in self.retrieve(query, domain=domain)]

    # ──────────────────────────────────────────────────────────────────────
    # Generation
    # ──────────────────────────────────────────────────────────────────────
    async def aquery(self, query: str) -> str:
        """Full RAG query — retrieval + generation."""
        if not self.query_engine:
            raise RuntimeError("Call setup_retrieval() first")
        response = await self.query_engine.aquery(query)
        return str(response)

    def query(self, query: str) -> str:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.aquery(query))

    # ──────────────────────────────────────────────────────────────────────
    # Dense-only ablation (for comparison experiments)
    # ──────────────────────────────────────────────────────────────────────
    def retrieve_dense_only(self, query: str, domain: str = "general") -> List[str]:
        search_query = self.reformulate_query(query, domain)
        r_cfg = self.config["retrieval"]
        dense = self.vector_index.as_retriever(similarity_top_k=r_cfg["dense_top_k"])
        loop = asyncio.get_event_loop()
        nodes = loop.run_until_complete(dense.aretrieve(search_query))
        return [n.node.get_content() for n in nodes]

    def retrieve_bm25_only(self, query: str, domain: str = "general") -> List[str]:
        search_query = self.reformulate_query(query, domain)
        loop = asyncio.get_event_loop()
        nodes = loop.run_until_complete(self.bm25_retriever.aretrieve(search_query))
        return [n.node.get_content() for n in nodes]
