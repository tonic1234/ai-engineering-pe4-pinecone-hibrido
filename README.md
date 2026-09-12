# Sistema RAG escalable en la nube con Pinecone

Pre-entrega 4 del curso **AI Engineering** (Coderhouse).
Ingesta a **Pinecone Serverless** con metadata, **recuperador híbrido** (BM25 + vectorial)
y evaluación con **Precision@k / Recall@k** sobre un *golden set*.

## Qué hay adentro

| Archivo | Qué hace |
|---|---|
| `setup_index.py` | Verifica si el índice existe y lo crea en modo Serverless si no está. |
| `ingest.py` | Carga el dataset, lo chunkea y lo sube a Pinecone con metadata. |
| `retriever.py` | `RAGSystem`: `EnsembleRetriever` (BM25 + vectorial), top-5. |
| `evaluate.py` | Corre el golden set y calcula Precision@5 y Recall@5. |
| `data/documentacion.json` | Dataset técnico (14 fragmentos con metadata: sección y categoría). |
| `golden_set.json` | 8 preguntas con la fuente esperada. |
| `tests/` | Pruebas del chunking con metadata y de las métricas. |

## Cómo correrlo

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # completar OPENAI_API_KEY y PINECONE_API_KEY

python setup_index.py       # crea el índice (1536 dims, métrica coseno)
python ingest.py            # sube los chunks con metadata
python evaluate.py          # imprime las métricas del golden set
```

> El recuperador funciona **aunque no haya Pinecone configurado**: en ese caso cae a modo
> léxico (solo BM25) y lo avisa por log. Sirve para probar el pipeline y las métricas sin
> depender de la nube.

## Variables de entorno

| Variable | Descripción |
|---|---|
| `OPENAI_API_KEY` | Para los embeddings (`text-embedding-3-small`, 1536 dims). |
| `PINECONE_API_KEY` | Requerida para el modo nube. |
| `INDEX_NAME` | Nombre del índice (por defecto `rag-tecnico`). |
| `PINECONE_NAMESPACE` | Namespace dentro del índice (por defecto `documentacion`). |
| `PINECONE_REGION` | Región del índice Serverless (por defecto `us-east-1`). |

## Ejemplo de salida de `evaluate.py`

```
=== Evaluación del recuperador (k=5) ===
Preguntas en el golden set : 8
Recall@5                 : 0.88
Precision@5              : 0.20

Detalle por pregunta:
  [OK ] ¿Qué pasa si llamo a time.sleep dentro de una función async? -> ['documentacion']
  ...
```

Es normal que la precisión sea baja: al traer 5 fragmentos y ser 1 el correcto, la
precisión es 1/5 = 20 %. Lo que se mira es que el **recall** sea alto.

## Decisiones de diseño

- **Metadata avanzada**: cada vector guarda `fuente`, `seccion`, `categoria` y también el
  **texto original**, para no tener que ir a buscar el contenido a otra base al recuperar.
- **Dimensiones**: el índice se crea con 1536 dims para que coincida con
  `text-embedding-3-small`. Un mismatch de dimensiones hace fallar el upsert.
- **Namespaces**: se usa un namespace explícito en vez del default, para que la búsqueda
  no quede mezclada con otros conjuntos de datos.
- **Recuperador híbrido**: BM25 acierta con términos exactos (siglas, nombres propios) donde
  el vector falla; el vector capta el significado donde BM25 no. `EnsembleRetriever` los
  combina con pesos 50/50.
- **Chunking**: 800 caracteres con 100 de solape (el punto medio que recomienda la consigna).

## Tests

```bash
pytest -q
```

Las métricas se validan contra un sistema falso con resultados calculados a mano, así el
test no depende de la nube.
