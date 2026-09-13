# Sistema RAG escalable en la nube con Pinecone

Pre-entrega 4 del curso **AI Engineering** (Coderhouse).
Ingesta a **Pinecone Serverless** con metadata, **recuperador híbrido** (BM25 + vectorial)
y evaluación con **Precision@5 / Recall@5** sobre un *golden set*.

## Contenido del repo

| Archivo | Qué hace |
|---|---|
| `setup_index.py` | Verifica si el índice existe y lo crea en modo Serverless (dimensión 384). |
| `ingest.py` | Carga el dataset, lo chunkea en tokens y lo sube a Pinecone con metadata. |
| `retriever.py` | `RAGSystem`: `EnsembleRetriever` (BM25 + vectorial) y `obtener_top_k()`. |
| `evaluate.py` | Golden set + `evaluar()` con Precision@5 y Recall@5. |
| `data/` | 4 políticas internas de ejemplo (.txt), tomando como dominio el de la empresa donde trabajo. |
| `tests/` | Pruebas del chunking con metadata y de las métricas. |

## Contexto del dataset

Uso como dominio de ejemplo el de **Dendra**, la agencia donde trabajo: son documentos del
tipo que existen en cualquier empresa (vacaciones, teletrabajo, seguridad, onboarding), y
eso le da a la búsqueda híbrida siglas y términos exactos reales para probar (2FA, MDM,
BYOD).

**Importante**: el contenido de los `.txt` es **inventado** — solo imita la lógica de una
política interna. No son las políticas reales de la empresa ni contienen datos reales de
nadie.

## Cómo correrlo

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # completar PINECONE_API_KEY

python setup_index.py       # crea el índice (384 dims, métrica coseno)
python ingest.py            # sube los chunks con metadata
python evaluate.py          # imprime Recall@5 y Precision@5
```

> El recuperador funciona **aunque no haya Pinecone configurado**: en ese caso cae a modo
> léxico (solo BM25) y lo avisa por log. Sirve para probar el pipeline y las métricas sin
> depender de la nube.

## Variables de entorno

| Variable | Descripción |
|---|---|
| `PINECONE_API_KEY` | Requerida para el modo nube (free tier). |
| `INDEX_NAME` | Nombre del índice (por defecto `dendra-rag-hibrido`). |
| `PINECONE_NAMESPACE` | Namespace dentro del índice (por defecto `politicas-dendra`). |
| `PINECONE_REGION` | Región del índice Serverless (por defecto `us-east-1`). |

## Ejemplo de salida de `evaluate.py`

```
================================================================================
✅ ¿Cuántos días de vacaciones corresponden a un empleado con 6 años de antigüedad?
   Esperado: politica_vacaciones.txt | Recuperados: ['politica_vacaciones.txt', ...]
   Recall@5: 100% | Precision@5: 25%
...
================================================================================
📊 RECALL@5 PROMEDIO:    100.0%
📊 PRECISION@5 PROMEDIO: 25.0%
```

Es normal que la precisión sea baja: al traer 5 fragmentos y ser 1 el correcto, la
precisión es 1/5 = 20 %. Lo que se mira es que el **recall** sea alto.

## Criterios de diseño

- **Embeddings locales**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensiones). No
  requieren API key ni cuestan nada, y corren en la máquina.
- **Dimensión del índice**: la importo del módulo de ingesta en vez de escribirla a mano.
  1536 es la dimensión de los embeddings de OpenAI, no la de este modelo local; fijarla mal
  provoca el "mismatch de dimensiones" que menciona la consigna.
- **Metadata avanzada**: cada vector guarda `source` (nombre de archivo), `categoria`
  (derivada del nombre) y `chunk_id`.
- **Namespaces**: se usa un namespace explícito en vez del default, para que la búsqueda no
  quede mezclada con otros conjuntos de datos.
- **Recuperador híbrido**: BM25 acierta con términos exactos (siglas, códigos) donde el
  vector falla; el vector capta el significado donde BM25 no. `EnsembleRetriever` los
  combina con pesos 50/50.
- **Chunking**: 600 tokens con 100 de solape (dentro del rango 500-800 sugerido).

## Tests

```bash
pytest -q
```

Las métricas se validan contra un sistema falso con resultados calculados a mano, así el
test no depende de la nube.
