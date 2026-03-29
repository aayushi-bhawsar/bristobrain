"""
data/rag_store.py
ChromaDB-backed RAG store for BistroBrain.
Indexes POS summaries and inventory snapshots for retrieval-augmented generation.
"""
from __future__ import annotations

import json
import os
from datetime import date

import chromadb
import structlog
from chromadb.config import Settings

from data.schema import DailySummary, InventoryItem

log = structlog.get_logger(__name__)

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8001"))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "bistrobrain_pos")


class RAGStore:
    """
    Wraps ChromaDB to provide:
    - Index daily POS summaries as retrievable documents
    - Query relevant summaries by semantic search
    - Index inventory snapshots
    """

    def __init__(self) -> None:
        try:
            self._client = chromadb.HttpClient(
                host=CHROMA_HOST,
                port=CHROMA_PORT,
                settings=Settings(anonymized_telemetry=False),
            )
        except Exception:
            # Fall back to in-process ephemeral client for local dev
            log.warning("chroma_http_unavailable", fallback="in_process")
            self._client = chromadb.EphemeralClient(
                settings=Settings(anonymized_telemetry=False)
            )

        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    # ─────────────────────────────────────────
    # Indexing
    # ─────────────────────────────────────────

    def index_summaries(self, summaries: list[DailySummary]) -> None:
        """Convert daily summaries to natural-language docs and upsert into Chroma."""
        documents: list[str] = []
        metadatas: list[dict] = []
        ids: list[str] = []

        for s in summaries:
            doc = (
                f"Date: {s.date}. "
                f"Revenue: {s.total_revenue}. Food cost: {s.total_food_cost}. "
                f"Gross profit: {s.gross_profit}. Avg margin: {s.avg_margin:.1%}. "
                f"Covers served: {s.cover_count}. "
                f"Top items: {', '.join(s.top_items)}. "
                f"Low-margin items: {', '.join(s.low_margin_items) or 'none'}."
            )
            documents.append(doc)
            metadatas.append(
                {
                    "date": str(s.date),
                    "total_revenue": s.total_revenue,
                    "avg_margin": s.avg_margin,
                    "cover_count": s.cover_count,
                    "top_items": json.dumps(s.top_items),
                }
            )
            ids.append(f"daily_{s.date}")

        self._collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
        log.info("summaries_indexed", count=len(summaries))

    def index_inventory(self, items: list[InventoryItem]) -> None:
        """Index current inventory snapshot into Chroma."""
        documents: list[str] = []
        metadatas: list[dict] = []
        ids: list[str] = []

        for item in items:
            doc = (
                f"Inventory item: {item.item_name} ({item.category.value}). "
                f"Stock: {item.quantity_kg}kg. Unit cost: {item.unit_cost}. "
                f"Expiry: {item.expiry_date or 'N/A'}. "
                f"Low stock: {'yes' if item.is_low_stock else 'no'}."
            )
            documents.append(doc)
            metadatas.append(
                {
                    "item_id": item.item_id,
                    "item_name": item.item_name,
                    "quantity_kg": item.quantity_kg,
                    "expiry_date": str(item.expiry_date) if item.expiry_date else "",
                    "is_low_stock": item.is_low_stock,
                }
            )
            ids.append(f"inv_{item.item_id}")

        self._collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
        log.info("inventory_indexed", count=len(items))

    # ─────────────────────────────────────────
    # Retrieval
    # ─────────────────────────────────────────

    def query(self, question: str, n_results: int = 5) -> list[str]:
        """
        Semantic search against the indexed POS + inventory documents.
        Returns a list of relevant text chunks for the LLM context.
        """
        results = self._collection.query(
            query_texts=[question],
            n_results=n_results,
        )
        docs: list[str] = results.get("documents", [[]])[0]
        log.info("rag_query", question=question[:80], results=len(docs))
        return docs

    def get_recent_summaries(self, days: int = 7) -> list[str]:
        """Retrieve the most recent N days of summaries (by metadata date)."""
        results = self._collection.get(
            where={"date": {"$gte": str(date.today())}},
            limit=days,
        )
        return results.get("documents", [])
