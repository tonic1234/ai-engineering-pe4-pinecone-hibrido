"""retriever.py — Recuperador híbrido: búsqueda vectorial (Pinecone) + léxica (BM25).

APUNTE DE CLASE (lo más interesante de esta pre-entrega):
La búsqueda semántica es muy buena entendiendo el significado, pero se le escapan los
términos exactos: nombres de funciones, siglas, versiones. BM25 es lo contrario: es
búsqueda clásica por palabras, y acierta justo ahí donde el vector falla.

La solución es combinarlas con un EnsembleRetriever, que hace un ranking combinado (por
defecto, 50 y 50). Es el mismo patrón que usa cualquier buscador serio.

Detalle de diseño: el texto original del chunk lo guardo EN LA METADATA de Pinecone.
Así, cuando recupero un vector, ya tengo el texto para el prompt y no necesito ir a
buscar a otra base.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

TOP_K = 5  # la consigna pide devolver los top-5


class RAGSystem:
    """Encapsula el recuperador híbrido y expone una única operación de búsqueda."""

    def __init__(self, documentos: list[Document], weights: tuple[float, float] = (0.5, 0.5)) -> None:
        if not documentos:
            raise ValueError("Hace falta al menos un documento para armar el retriever")

        self.documentos = documentos
        self.weights = weights
        self._retriever = self._build()

    def _build(self):
        """Arma el EnsembleRetriever: BM25 + (vectorial si hay Pinecone)."""

        # Rama léxica: siempre disponible, corre local.
        bm25 = BM25Retriever.from_documents(self.documentos, k=TOP_K)

        # Rama vectorial: solo si hay credenciales de Pinecone.
        vector = None
        if os.getenv("PINECONE_API_KEY"):
            try:
                from ingest import build_vector_retriever

                vector = build_vector_retriever(k=TOP_K)
            except Exception as exc:  # si algo falla, seguimos solo con BM25
                logger.warning("No pude armar la rama vectorial (%s); uso solo BM25", exc)

        if vector is None:
            logger.info("Recuperador en modo LÉXICO (sin Pinecone configurado)")
            return bm25

        from langchain.retrievers import EnsembleRetriever

        logger.info("Recuperador HÍBRIDO (BM25 + vectorial) con pesos %s", self.weights)
        return EnsembleRetriever(retrievers=[bm25, vector], weights=list(self.weights))

    def search(self, query: str, k: int = TOP_K) -> list[Document]:
        """Devuelve los top-k documentos combinando lo léxico y lo semántico."""

        resultados = self._retriever.invoke(query)
        return resultados[:k]

    def as_langchain(self) -> Any:
        """Devuelve el retriever para poder usarlo dentro de una cadena LCEL."""

        return self._retriever


def load_chunks() -> list[Document]:
    """Carga y parte documentos técnicos para poblar el índice."""

    from ingest import chunk_documents, load_dataset

    return chunk_documents(load_dataset())
