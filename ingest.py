"""ingest.py — Ingesta de documentos hacia Pinecone con metadata.

Nota:
Acá el punto es no subir texto pelado. Cada vector viaja con metadata: de dónde salió
(fuente), su categoría y el índice del chunk. Esa metadata después se usa para FILTRAR la
búsqueda y para armar las fuentes de la respuesta sin que el LLM tenga que "recordarlas".

El chunking lo hago con from_tiktoken_encoder para medir en TOKENS (600 con 100 de
solape), que es el rango 500-800 que recomienda la consigna.

Sobre los embeddings: uso un modelo LOCAL de HuggingFace (all-MiniLM-L6-v2), que tiene
384 dimensiones. Antes de esto hay que fijar la dimensión del índice: si subo vectores de
384 a un índice creado con 1536, Pinecone rechaza el upsert (es el "mismatch de
dimensiones" que menciona la consigna).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

logger = logging.getLogger(__name__)

DATA_DIR = Path("./data")
CHUNK_SIZE = 600      # tokens (rango sugerido 500-800)
CHUNK_OVERLAP = 100

# Modelo de embeddings local y gratuito. Su dimensión es 384.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

INDEX_NAME = os.getenv("INDEX_NAME", "dendra-rag-hibrido")
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "politicas-dendra")


def load_dataset(data_dir: Path = DATA_DIR) -> list[Document]:
    """Carga todos los .txt de /data como Documents."""

    loader = DirectoryLoader(
        str(data_dir),
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documentos = loader.load()
    if not documentos:
        raise FileNotFoundError(f"No encontré documentos en {data_dir.resolve()}")
    logger.info("Cargué %d documento(s)", len(documentos))
    return documentos


def get_embeddings() -> HuggingFaceEmbeddings:
    """Un único lugar donde se define el modelo de embeddings.

    Regla de oro: indexar y consultar SIEMPRE con el mismo modelo.
    """

    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def chunk_documents(documentos: list[Document]) -> list[Document]:
    """Parte los documentos en chunks medidos en tokens y agrega metadata."""

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documentos)

    # Metadata avanzada: nombre de archivo limpio, categoría derivada e índice del chunk.
    for i, chunk in enumerate(chunks):
        nombre_archivo = os.path.basename(chunk.metadata.get("source", "documento.txt"))
        chunk.metadata["source"] = nombre_archivo
        chunk.metadata["categoria"] = nombre_archivo.replace(".txt", "").replace("_", " ")
        chunk.metadata["chunk_id"] = i

    logger.info("✂️  Fragmentos generados: %d", len(chunks))
    return chunks


def build_vectorstore():
    """Sube los chunks a Pinecone y devuelve el vector store (o None si no hay credenciales)."""

    if not os.getenv("PINECONE_API_KEY"):
        logger.warning("Sin PINECONE_API_KEY: no se puede armar el vector store")
        return None

    from langchain_pinecone import PineconeVectorStore

    chunks = chunk_documents(load_dataset())
    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        index_name=INDEX_NAME,
        namespace=NAMESPACE,
    )
    logger.info("📦 Chunks subidos a Pinecone: %d", len(chunks))
    return vectorstore


def build_vector_retriever(k: int = 5):
    """Retriever vectorial (Pinecone) para combinar con BM25."""

    from langchain_pinecone import PineconeVectorStore

    store = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=get_embeddings(),
        namespace=NAMESPACE,
    )
    return store.as_retriever(search_kwargs={"k": k, "namespace": NAMESPACE})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from setup_index import ensure_index

    ensure_index()
    build_vectorstore()
