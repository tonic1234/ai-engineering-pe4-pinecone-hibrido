"""setup_index.py — Crea (si hace falta) el índice Serverless de Pinecone.

APUNTE DE CLASE:
Dos cosas que hay que cuidar acá y que son las que más errores causan:
  1. La DIMENSIÓN del índice tiene que coincidir con la del modelo de embeddings. Como
     ahora uso un modelo local (all-MiniLM-L6-v2), la dimensión es 384, NO 1536 (ese
     número es específico de los embeddings de OpenAI). Por eso la dimensión la importo
     del módulo de ingesta en vez de escribirla a mano: si cambio el modelo de embeddings,
     se cambia sola.
  2. El NAMESPACE: es como una "carpeta" dentro del índice. Si mezclo todo en el namespace
     por defecto, la búsqueda se vuelve ruidosa.

El script es idempotente a propósito: si el índice ya existe, no lo toca.
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

from ingest import EMBEDDING_DIM, INDEX_NAME, NAMESPACE

load_dotenv()

logger = logging.getLogger(__name__)


def get_client():
    """Devuelve un cliente de Pinecone (o None si no hay credenciales)."""

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        logger.warning("Sin PINECONE_API_KEY: el módulo va a correr en modo local (solo BM25)")
        return None

    from pinecone import Pinecone

    return Pinecone(api_key=api_key)


def ensure_index(index_name: str | None = None, dimension: int = EMBEDDING_DIM):
    """Verifica si el índice existe y lo crea en modo Serverless si no está."""

    index_name = index_name or INDEX_NAME
    pc = get_client()
    if pc is None:
        return None

    from pinecone import ServerlessSpec

    indices_existentes = [i["name"] for i in pc.list_indexes()]

    if index_name not in indices_existentes:
        logger.info("🆕 Creando índice '%s' (dimensión %d)...", index_name, dimension)
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=os.getenv("PINECONE_REGION", "us-east-1")),
        )
    else:
        logger.info("♻️  El índice '%s' ya existe — no se vuelve a crear", index_name)

    indice = pc.Index(index_name)
    print(f"Índice {index_name} listo (namespace: {NAMESPACE})")
    return indice


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    indice = ensure_index()
    print("Índice listo:", indice is not None)
