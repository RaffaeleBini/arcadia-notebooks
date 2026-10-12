"""
retrieval.py — Pipeline de consulta: embed → retrieve → rewrite → generate.
"""

from dataclasses import dataclass, field
from openai import OpenAI
import chromadb

MODELO_EMBEDDING  = "text-embedding-3-small"
MODELO_GENERACION = "gpt-4o-mini"
UMBRAL_RELEVANCIA = 0.65

SYSTEM_RAG = """You are a documentation assistant. Answer questions using ONLY the provided context.
If the answer is not in the context, respond EXACTLY with: INSUFFICIENT_CONTEXT
Never use external knowledge. Be concise. Always mention the source at the end."""

SYSTEM_REWRITER = """Rewrite the following question as an optimized semantic search query.
Make it specific and technical. Return only the rewritten query, no explanation."""


@dataclass
class ChunkRecuperado:
    texto:     str
    similitud: float
    fuente:    str
    categoria: str


@dataclass
class ResultadoRetrieval:
    pregunta:          str
    query_retrieval:   str
    chunks:            list[ChunkRecuperado]
    respondida:        bool
    respuesta:         str
    latencia_ms:       float = 0.0


def generar_embedding(texto: str, cliente: OpenAI) -> list[float]:
    r = cliente.embeddings.create(model=MODELO_EMBEDDING, input=texto)
    return r.data[0].embedding


def reescribir_query(pregunta: str, cliente: OpenAI) -> str:
    r = cliente.chat.completions.create(
        model=MODELO_GENERACION,
        messages=[
            {"role": "system",  "content": SYSTEM_REWRITER},
            {"role": "user",    "content": pregunta}
        ],
        temperature=0.0, max_tokens=100
    )
    return r.choices[0].message.content.strip()


def recuperar(
    query: str,
    coleccion: chromadb.Collection,
    cliente: OpenAI,
    k: int = 4,
    filtro: dict | None = None
) -> list[ChunkRecuperado]:
    """Embeddea la query y recupera los k chunks más similares."""
    vector = generar_embedding(query, cliente)
    kwargs = {
        "query_embeddings": [vector],
        "n_results":        min(k, coleccion.count()),
        "include":          ["documents", "distances", "metadatas"]
    }
    if filtro:
        kwargs["where"] = filtro

    resp = coleccion.query(**kwargs)
    return [
        ChunkRecuperado(
            texto=doc,
            similitud=round(1 - dist, 4),
            fuente=meta.get("fuente", "desconocida"),
            categoria=meta.get("categoria", "general")
        )
        for doc, dist, meta in zip(
            resp["documents"][0], resp["distances"][0], resp["metadatas"][0]
        )
    ]


def construir_prompt(pregunta: str, chunks: list[ChunkRecuperado]) -> str:
    bloques = [
        f"[Context {i} | Source: {c.fuente} | Score: {c.similitud:.2f}]\n{c.texto}"
        for i, c in enumerate(chunks, 1)
    ]
    return f"CONTEXT:\n{'\\n\\n---\\n\\n'.join(bloques)}\\n\\nQUESTION: {pregunta}\\n\\nANSWER:"


def generar_respuesta(prompt: str, historial: list[dict], cliente: OpenAI) -> str:
    """Genera la respuesta final usando el historial de conversación como contexto."""
    mensajes = (
        [{"role": "system", "content": SYSTEM_RAG}]
        + historial[-6:]   # máximo 3 turnos previos
        + [{"role": "user", "content": prompt}]
    )
    r = cliente.chat.completions.create(
        model=MODELO_GENERACION, messages=mensajes,
        temperature=0.1, max_tokens=600
    )
    return r.choices[0].message.content


def pipeline_consulta(
    pregunta: str,
    coleccion: chromadb.Collection,
    cliente: OpenAI,
    historial: list[dict] | None = None,
    k: int = 4,
    umbral: float = UMBRAL_RELEVANCIA,
    usar_rewriting: bool = True
) -> ResultadoRetrieval:
    """
    Pipeline completo de consulta RAG.
    1. Reescribir query (opcional)
    2. Recuperar chunks
    3. Filtrar por umbral
    4. Generar respuesta
    """
    import time
    t = time.perf_counter()
    historial = historial or []

    query_retrieval = reescribir_query(pregunta, cliente) if usar_rewriting else pregunta

    chunks_todos      = recuperar(query_retrieval, coleccion, cliente, k=k)
    chunks_relevantes = [c for c in chunks_todos if c.similitud >= umbral]

    if not chunks_relevantes:
        latencia = (time.perf_counter() - t) * 1000
        return ResultadoRetrieval(
            pregunta=pregunta,
            query_retrieval=query_retrieval,
            chunks=[],
            respondida=False,
            respuesta="I don't have enough information in the available documentation to answer this question.",
            latencia_ms=round(latencia, 1)
        )

    prompt    = construir_prompt(pregunta, chunks_relevantes)
    respuesta = generar_respuesta(prompt, historial, cliente)
    respondida = "INSUFFICIENT_CONTEXT" not in respuesta

    if not respondida:
        respuesta = "I don't have enough information in the available documentation to answer this question."

    latencia = (time.perf_counter() - t) * 1000
    return ResultadoRetrieval(
        pregunta=pregunta,
        query_retrieval=query_retrieval,
        chunks=chunks_relevantes,
        respondida=respondida,
        respuesta=respuesta,
        latencia_ms=round(latencia, 1)
    )
