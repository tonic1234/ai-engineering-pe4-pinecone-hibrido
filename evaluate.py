"""evaluate.py — Evaluación del recuperador con un Golden Set.

APUNTE DE CLASE:
Para saber si un RAG funciona hay que medirlo, y la forma más simple es un "golden
set": un puñado de preguntas donde YO sé cuál es el documento que debería recuperarse.

Las dos métricas que pide la consigna:
  - Recall@k: de todas las preguntas, en cuántas el documento correcto apareció dentro
    de los k recuperados. Mide si NO me estoy perdiendo la respuesta.
  - Precision@k: de los k documentos que devolví, qué proporción era realmente útil.
    Mide si no estoy ensuciando el contexto con ruido.

Es normal que el recall sea más alto que la precisión: traer 5 documentos y que 1 sea
el correcto da recall 100% y precisión 20%.

Uso:  python evaluate.py     (funciona aunque no haya Pinecone: cae a modo léxico)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from retriever import TOP_K, RAGSystem, load_chunks

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

GOLDEN_SET = Path("./golden_set.json")


def load_golden_set(path: Path = GOLDEN_SET) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Falta el golden set en {path.resolve()}")
    return json.loads(path.read_text(encoding="utf-8"))


def _doc_key(doc) -> str:
    """Identificador del documento para comparar contra lo esperado."""

    meta = doc.metadata or {}
    return str(meta.get("fuente") or meta.get("source") or "")


def evaluate(system: RAGSystem, golden: list[dict], k: int = TOP_K) -> dict:
    """Calcula Precision@k y Recall@k sobre el golden set."""

    recall_hits = 0
    precision_scores: list[float] = []
    detalle: list[dict] = []

    for caso in golden:
        pregunta = caso["pregunta"]
        esperado = caso["fuente_esperada"]

        recuperados = system.search(pregunta, k=k)
        fuentes = [_doc_key(d) for d in recuperados]

        acierto = esperado in fuentes
        recall_hits += int(acierto)

        utiles = sum(1 for f in fuentes if f == esperado)
        precision = utiles / len(fuentes) if fuentes else 0.0
        precision_scores.append(precision)

        detalle.append(
            {
                "pregunta": pregunta,
                "esperado": esperado,
                "recuperados": fuentes,
                "recall@k": acierto,
                "precision@k": round(precision, 2),
            }
        )

    return {
        "k": k,
        "n_preguntas": len(golden),
        "recall@k": round(recall_hits / len(golden), 2) if golden else 0.0,
        "precision@k": round(sum(precision_scores) / len(precision_scores), 2) if precision_scores else 0.0,
        "detalle": detalle,
    }


def main() -> None:
    chunks = load_chunks()
    system = RAGSystem(chunks)
    resultados = evaluate(system, load_golden_set())

    print(f"\n=== Evaluación del recuperador (k={resultados['k']}) ===")
    print(f"Preguntas en el golden set : {resultados['n_preguntas']}")
    print(f"Recall@{resultados['k']}                 : {resultados['recall@k']}")
    print(f"Precision@{resultados['k']}              : {resultados['precision@k']}")
    print("\nDetalle por pregunta:")
    for fila in resultados["detalle"]:
        estado = "OK " if fila["recall@k"] else "FALLO"
        print(f"  [{estado}] {fila['pregunta'][:55]:<55} -> {fila['recuperados']}")


if __name__ == "__main__":
    main()
