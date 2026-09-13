"""tests/test_retriever.py — Pruebas del recuperador y de la evaluación.

APUNTE: BM25 corre 100% local, así que puedo probar el recuperador y las métricas sin
Pinecone y sin API key. Lo que verifico:
  1. Que el chunking conserve la metadata (fuente/categoría) — es lo que después permite
     comparar contra el golden set.
  2. Que el recuperador devuelva como máximo k documentos.
  3. Que Recall@k y Precision@k den los valores que calculo a mano (métricas confiables).

Correr:  pytest -q
"""

from __future__ import annotations

from evaluate import GOLDEN_SET, evaluar
from ingest import CHUNK_OVERLAP, CHUNK_SIZE, chunk_documents, load_dataset
from retriever import TOP_K, RAGSystem


def test_carga_y_chunkea():
    chunks = chunk_documents(load_dataset())
    assert len(chunks) >= 4
    assert all("source" in c.metadata for c in chunks)


def test_metadata_avanzada_en_chunks():
    chunks = chunk_documents(load_dataset())
    # La metadata que agrega el pipeline tiene que sobrevivir al split.
    assert all("categoria" in c.metadata and "chunk_id" in c.metadata for c in chunks)
    # Y el nombre de archivo no debe quedar con la ruta completa.
    assert all("/" not in c.metadata["source"] for c in chunks)


def test_chunking_configuracion_en_tokens():
    assert 500 <= CHUNK_SIZE <= 800  # rango sugerido por la consigna
    assert CHUNK_OVERLAP > 0


def test_busqueda_devuelve_como_maximo_k():
    sistema = RAGSystem(k=TOP_K)
    resultados = sistema.obtener_top_k("¿Cuántos días de vacaciones tengo?")
    assert 1 <= len(resultados) <= TOP_K
    assert {"contenido", "fuente", "categoria"} <= set(resultados[0])


def test_busqueda_lexica_encuentra_termino_exacto():
    sistema = RAGSystem(k=TOP_K)
    resultados = sistema.obtener_top_k("2FA MDM contraseñas")
    texto = " ".join(r["contenido"] for r in resultados).lower()
    assert "2fa" in texto or "mdm" in texto or "contraseñas" in texto


def test_golden_set_apunta_a_documentos_reales():
    # El golden set usa el nombre de archivo; el pipeline lo normaliza en chunk_documents.
    fuentes = {c.metadata["source"] for c in chunk_documents(load_dataset())}
    for caso in GOLDEN_SET:
        assert caso["documento_id_esperado"] in fuentes


def test_metricas_calculadas_a_mano():
    """Con un sistema falso controlado, las métricas tienen que dar exacto."""

    class SistemaFalso:
        def __init__(self):
            self.k = 2

        def obtener_top_k(self, query, k=None):
            # Para la primera pregunta acierta; para la segunda falla.
            if "uno" in query:
                return [{"fuente": "a"}, {"fuente": "b"}]
            return [{"fuente": "c"}, {"fuente": "d"}]

    golden = [
        {"pregunta": "pregunta uno", "documento_id_esperado": "a"},
        {"pregunta": "pregunta dos", "documento_id_esperado": "a"},
    ]
    resultado = evaluar(SistemaFalso(), golden, k=2)

    assert resultado["recall@2_promedio"] == 0.5      # 1 de 2 aciertos
    assert resultado["precision@2_promedio"] == 0.25  # (1/2 + 0/2) / 2
    assert len(resultado["detalle"]) == 2
