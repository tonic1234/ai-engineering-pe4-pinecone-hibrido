"""evaluate.py — Evaluación del recuperador con un Golden Set.

APUNTE DE CLASE:
Para saber si un RAG funciona hay que medirlo, y la forma más simple es un "golden set":
un puñado de preguntas donde YO sé cuál es el documento que debería recuperarse.

Las dos métricas que pide la consigna:
  - Recall@5: de todas las preguntas, en cuántas el documento correcto apareció dentro de
    los 5 recuperados. Mide si NO me estoy perdiendo la respuesta.
  - Precision@5: de los 5 documentos que devolví, qué proporción era realmente útil.
    Mide si no estoy ensuciando el contexto con ruido.

Es normal que el recall sea más alto que la precisión: traer 5 documentos y que 1 sea el
correcto da recall 100% y precisión 20%.

Uso:  python evaluate.py     (funciona aunque no haya Pinecone: cae a modo léxico)
"""

from __future__ import annotations

import logging
from typing import Dict, List

from retriever import RAGSystem

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# Golden set: preguntas donde conozco de antemano el documento fuente.
GOLDEN_SET = [
    {
        "pregunta": "¿Cuántos días de vacaciones corresponden a un empleado con 6 años de antigüedad?",
        "documento_id_esperado": "politica_vacaciones.txt",
    },
    {
        "pregunta": "¿Cuántos días de trabajo remoto por semana tiene el esquema estándar?",
        "documento_id_esperado": "politica_teletrabajo.txt",
    },
    {
        "pregunta": "¿Cada cuánto deben renovarse las contraseñas corporativas?",
        "documento_id_esperado": "politica_seguridad_informatica.txt",
    },
    {
        "pregunta": "¿Cuánto dura el proceso de onboarding?",
        "documento_id_esperado": "onboarding_nuevos_empleados.txt",
    },
    {
        "pregunta": "¿Qué esquema de trabajo tienen las áreas de Soporte Técnico Nivel 1 y Recepción?",
        "documento_id_esperado": "politica_teletrabajo.txt",
    },
]


def evaluar(rag_system: RAGSystem, golden_set: List[Dict] | None = None, k: int = 5) -> Dict:
    """Calcula Precision@k y Recall@k sobre el golden set."""

    golden_set = golden_set or GOLDEN_SET
    resultados_por_pregunta = []

    for caso in golden_set:
        top_k = rag_system.obtener_top_k(caso["pregunta"])
        fuentes_recuperadas = [r["fuente"] for r in top_k]

        # Recall@k: ¿el documento esperado aparece entre los recuperados? (0 o 1)
        recall = 1.0 if caso["documento_id_esperado"] in fuentes_recuperadas else 0.0

        # Precision@k: qué % de los recuperados son útiles (coinciden con la fuente esperada)
        coincidencias = sum(1 for f in fuentes_recuperadas if f == caso["documento_id_esperado"])
        precision = coincidencias / len(fuentes_recuperadas) if fuentes_recuperadas else 0.0

        resultados_por_pregunta.append(
            {
                "pregunta": caso["pregunta"],
                "esperado": caso["documento_id_esperado"],
                "recuperados": fuentes_recuperadas,
                f"recall@{k}": recall,
                f"precision@{k}": precision,
            }
        )

    recall_promedio = (
        sum(r[f"recall@{k}"] for r in resultados_por_pregunta) / len(resultados_por_pregunta)
        if resultados_por_pregunta
        else 0.0
    )
    precision_promedio = (
        sum(r[f"precision@{k}"] for r in resultados_por_pregunta) / len(resultados_por_pregunta)
        if resultados_por_pregunta
        else 0.0
    )

    return {
        "detalle": resultados_por_pregunta,
        f"recall@{k}_promedio": recall_promedio,
        f"precision@{k}_promedio": precision_promedio,
    }


def main() -> None:
    sistema = RAGSystem()
    reporte = evaluar(sistema, GOLDEN_SET, k=sistema.k)

    print("=" * 80)
    for r in reporte["detalle"]:
        estado = "✅" if r["recall@5"] == 1.0 else "❌"
        print(f"{estado} {r['pregunta']}")
        print(f"   Esperado: {r['esperado']} | Recuperados: {r['recuperados']}")
        print(f"   Recall@5: {r['recall@5']:.0%} | Precision@5: {r['precision@5']:.0%}\n")
    print("=" * 80)
    print(f"📊 RECALL@5 PROMEDIO:    {reporte['recall@5_promedio']:.1%}")
    print(f"📊 PRECISION@5 PROMEDIO: {reporte['precision@5_promedio']:.1%}")


if __name__ == "__main__":
    main()
