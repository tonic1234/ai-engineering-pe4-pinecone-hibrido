"""tests/test_retriever.py — Pruebas del recuperador y de la evaluación.

APUNTE: BM25 corre 100% local, así que puedo probar el recuperador y las métricas sin
Pinecone y sin API key. Lo que verifico:
  1. Que el chunking conserve la metadata (fuente/categoría) — es lo que después permite
     filtrar y comparar contra el golden set.
  2. Que el recuperador devuelva como máximo k documentos.
  3. Que Recall@k y Precision@k den los valores que calculo a mano (métricas confiables).

Correr:  pytest -q
"""

from __future__ import annotations

import pytest

from evaluate import evaluate
from ingest import chunk_documents, load_dataset
from retriever import TOP_K, RAGSystem


@pytest.fixture(scope="module")
def chunks():
    return chunk_documents(load_dataset())


@pytest.fixture(scope="module")
def sistema(chunks):
    return RAGSystem(chunks)


def test_carga_y_chunkea(chunks):
    assert len(chunks) >= 14  # el dataset tiene 14 entradas
    assert all("fuente" in c.metadata for c in chunks)


def test_metadata_preservada_en_chunks(chunks):
    # La metadata de la entrada original tiene que sobrevivir al split.
    assert all("categoria" in c.metadata for c in chunks)


def test_busqueda_devuelve_como_maximo_k(sistema):
    resultados = sistema.search("¿Qué es el event loop?", k=TOP_K)
    assert 1 <= len(resultados) <= TOP_K


def test_busqueda_lexica_encuentra_termino_exacto(sistema):
    # BM25 debería acertar con una sigla o nombre exacto.
    resultados = sistema.search("SqliteSaver thread_id", k=TOP_K)
    texto = " ".join(d.page_content for d in resultados).lower()
    assert "sqlitesaver" in texto or "checkpointer" in texto


def test_metricas_calculadas_a_mano():
    """Con un sistema falso controlado, las métricas tienen que dar exacto."""

    class Doc:
        def __init__(self, fuente):
            self.metadata = {"fuente": fuente}
            self.page_content = "texto"

    class SistemaFalso:
        def search(self, query, k=TOP_K):
            # Para la primera pregunta acierta; para la segunda falla.
            if "uno" in query:
                return [Doc("a"), Doc("b")]
            return [Doc("c"), Doc("d")]

    golden = [
        {"pregunta": "pregunta uno", "fuente_esperada": "a"},
        {"pregunta": "pregunta dos", "fuente_esperada": "a"},
    ]
    resultado = evaluate(SistemaFalso(), golden, k=2)

    assert resultado["recall@k"] == 0.5         # 1 de 2 aciertos
    assert resultado["precision@k"] == 0.25     # (1/2 + 0/2) / 2
    assert resultado["n_preguntas"] == 2
