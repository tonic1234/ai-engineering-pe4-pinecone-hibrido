"""retriever.py — Recuperador híbrido: búsqueda vectorial (Pinecone) + léxica (BM25).

APUNTE DE CLASE (lo más interesante de esta pre-entrega):
La búsqueda semántica es muy buena entendiendo el significado ("¿cuánto puedo descansar
por año?" ≈ "vacaciones"), pero se le escapan los términos exactos: siglas, nombres
propios, códigos. BM25 es lo contrario: busca coincidencias de PALABRAS literales, y
acierta justo ahí donde el vector falla.

La solución es combinarlas con un EnsembleRetriever, que hace un ranking combinado (por
defecto, 50 y 50).

Detalle de diseño: el texto original del chunk lo guardo en la metadata del vector. Así,
cuando recupero un vector, ya tengo el texto para el prompt y no necesito ir a buscarlo a
otra base de datos.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List

from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from ingest import chunk_documents, load_dataset

# Cargo el .env para que PINECONE_API_KEY esté disponible si se corre el script solo.
load_dotenv()

logger = logging.getLogger(__name__)

TOP_K = 5  # la consigna pide devolver los top-5


def build_hybrid_retriever(k: int = TOP_K):
    """Arma el EnsembleRetriever si hay Pinecone; si no, devuelve solo BM25 (modo local)."""

    # Rama léxica: siempre disponible, corre local.
    chunks = chunk_documents(load_dataset())
    retriever_bm25 = BM25Retriever.from_documents(chunks)
    retriever_bm25.k = k

    # Rama vectorial: solo si hay credenciales de Pinecone.
    if os.getenv("PINECONE_API_KEY"):
        try:
            from ingest import build_vector_retriever

            retriever_vectorial = build_vector_retriever(k=k)
        except Exception as exc:  # si algo falla, seguimos solo con BM25
            logger.warning("No pude armar la rama vectorial (%s); uso solo BM25", exc)
            retriever_vectorial = None
    else:
        retriever_vectorial = None

    if retriever_vectorial is None:
        logger.info("Recuperador en modo LÉXICO (sin Pinecone configurado)")
        return retriever_bm25

    # El import de EnsembleRetriever vive en langchain-classic a partir de LangChain 1.x.
    from langchain_classic.retrievers import EnsembleRetriever

    logger.info("Recuperador HÍBRIDO (BM25 + vectorial) con pesos 50/50")
    return EnsembleRetriever(retrievers=[retriever_bm25, retriever_vectorial], weights=[0.5, 0.5])


class RAGSystem:
    """Encapsula el EnsembleRetriever y expone un método simple para obtener el top-k."""

    def __init__(self, retriever=None, k: int = TOP_K) -> None:
        self.retriever = retriever or build_hybrid_retriever(k)
        self.k = k

    def obtener_top_k(self, query: str) -> List[Dict]:
        """Devuelve los top-k documentos combinando lo léxico y lo semántico."""

        docs: List[Document] = self.retriever.invoke(query)[: self.k]
        return [
            {
                "contenido": d.page_content,
                "fuente": d.metadata.get("source", "desconocida"),
                "categoria": d.metadata.get("categoria", "desconocida"),
            }
            for d in docs
        ]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    sistema = RAGSystem()
    for i, r in enumerate(sistema.obtener_top_k("¿Cuántos días de vacaciones tengo con 6 años de antigüedad?"), 1):
        print(f"{i}. [{r['fuente']}] {r['contenido'][:120]}...")
