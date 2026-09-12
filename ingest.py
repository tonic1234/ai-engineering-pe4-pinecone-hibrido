"""ingest.py — Ingesta de documentos hacia Pinecone con metadata.

APUNTE DE CLASE:
Acá el punto es no subir texto pelado. Cada vector viaja con metadata: de dónde salió
(fuente), en qué página estaba y a qué categoría pertenece. Esa metadata después se usa
para FILTRAR la búsqueda (por ejemplo, buscar solo dentro de una categoría).

También dejo el texto original dentro de la metadata. Suena redundante, pero evita
tener que ir a buscar el contenido a otra base cada vez que recupero un fragmento.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

logger = logging.getLogger(__name__)

DATA_DIR = Path("./data")
CHUNK_SIZE = 800      # ~500-800 tokens, el punto medio que recomienda la consigna
CHUNK_OVERLAP = 100
BATCH_SIZE = 100


def load_dataset(data_dir: Path = DATA_DIR) -> list[Document]:
    """Carga los documentos del dataset (.md, .txt o .json) con su metadata."""

    documentos: list[Document] = []
    for archivo in sorted(data_dir.glob("*")):
        if archivo.suffix == ".json":
            data = json.loads(archivo.read_text(encoding="utf-8"))
            for i, item in enumerate(data):
                documentos.append(
                    Document(
                        page_content=item["texto"],
                        metadata={
                            "fuente": archivo.stem,
                            "seccion": item.get("seccion", f"item-{i}"),
                            "categoria": item.get("categoria", "general"),
                        },
                    )
                )
        elif archivo.suffix in {".md", ".txt"}:
            documentos.append(
                Document(
                    page_content=archivo.read_text(encoding="utf-8"),
                    metadata={"fuente": archivo.stem, "pagina": 1, "categoria": "documentacion"},
                )
            )

    if not documentos:
        raise FileNotFoundError(f"No encontré documentos en {data_dir.resolve()}")
    logger.info("Cargué %d documento(s)", len(documentos))
    return documentos


def chunk_documents(documentos: list[Document]) -> list[Document]:
    """Parte los documentos en chunks conservando la metadata."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )
    # add_start_index guarda la posición del chunk dentro del documento original.
    chunks = splitter.split_documents(documentos)
    logger.info("Generé %d chunks (size=%d, overlap=%d)", len(chunks), CHUNK_SIZE, CHUNK_OVERLAP)
    return chunks


def build_vector_retriever(k: int = 5):
    """Sube los chunks a Pinecone y devuelve un retriever vectorial de LangChain."""

    from langchain_openai import OpenAIEmbeddings
    from langchain_pinecone import PineconeVectorStore

    index_name = os.getenv("INDEX_NAME", "rag-tecnico")
    namespace = os.getenv("PINECONE_NAMESPACE", "documentacion")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    store = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        namespace=namespace,
    )
    return store.as_retriever(search_kwargs={"k": k})


def upload(chunks: list[Document] | None = None) -> int:
    """Sube los chunks a Pinecone por lotes. Devuelve cuántos subió."""

    from langchain_openai import OpenAIEmbeddings
    from langchain_pinecone import PineconeVectorStore

    chunks = chunks or chunk_documents(load_dataset())
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    store = PineconeVectorStore(
        index_name=os.getenv("INDEX_NAME", "rag-tecnico"),
        embedding=embeddings,
        namespace=os.getenv("PINECONE_NAMESPACE", "documentacion"),
    )

    for i in range(0, len(chunks), BATCH_SIZE):
        lote = chunks[i : i + BATCH_SIZE]
        store.add_documents(lote)
        logger.info("Subido lote %d (%d chunks)", i // BATCH_SIZE + 1, len(lote))

    return len(chunks)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from setup_index import ensure_index

    ensure_index()
    total = upload()
    print(f"Listo: {total} chunks indexados en Pinecone")
