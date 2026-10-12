"""
ingesta.py — Pipeline de carga, limpieza, chunking e indexado de documentos.
"""

import re
import datetime
import requests
from pathlib import Path
from openai import OpenAI
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _cliente_openai() -> OpenAI:
    import os
    from dotenv import load_dotenv
    load_dotenv()
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


MODELO_EMBEDDING = "text-embedding-3-small"
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""],
)


# ── Loaders ───────────────────────────────────────────────────────────────────

def cargar_wikipedia(titulo: str, idioma: str = "en") -> str:
    """Descarga el texto completo de un artículo de Wikipedia."""
    url    = f"https://{idioma}.wikipedia.org/w/api.php"
    params = {
        "action": "query", "format": "json",
        "titles": titulo, "prop": "extracts",
        "explaintext": True, "exsectionformat": "plain"
    }
    try:
        resp  = requests.get(url, params=params, timeout=10)
        pages = resp.json()["query"]["pages"]
        texto = list(pages.values())[0].get("extract", "")
        return texto
    except Exception as e:
        print(f"Error cargando '{titulo}': {e}")
        return ""


def cargar_markdown(ruta: str) -> str:
    """Lee un archivo Markdown desde disco."""
    return Path(ruta).read_text(encoding="utf-8")


def cargar_txt(ruta: str) -> str:
    """Lee un archivo de texto plano desde disco."""
    return Path(ruta).read_text(encoding="utf-8")


# ── Cleaner ───────────────────────────────────────────────────────────────────

def limpiar(texto: str) -> str:
    """Elimina ruido: caracteres de control, espacios múltiples, referencias Wikipedia."""
    texto = re.sub(r"[\x00-\x08\x0b\x0e-\x1f\x7f]", "", texto)
    texto = re.sub(r" +",      " ",    texto)
    texto = re.sub(r"\n{3,}",  "\n\n", texto)
    texto = re.sub(r"== References ==.*", "", texto, flags=re.DOTALL)
    texto = re.sub(r"== External links ==.*", "", texto, flags=re.DOTALL)
    return texto.strip()


# ── Chunker ───────────────────────────────────────────────────────────────────

def chunkear(texto: str) -> list[str]:
    """Divide el texto en chunks con RecursiveCharacterTextSplitter."""
    return [c for c in _splitter.split_text(texto) if len(c) > 100]


# ── Embedder ──────────────────────────────────────────────────────────────────

def embeddear_batch(textos: list[str], cliente: OpenAI) -> list[list[float]]:
    """Genera embeddings para un lote de textos."""
    r = cliente.embeddings.create(model=MODELO_EMBEDDING, input=textos)
    return [item.embedding for item in sorted(r.data, key=lambda x: x.index)]


# ── Indexer ───────────────────────────────────────────────────────────────────

def indexar(
    coleccion: chromadb.Collection,
    chunks: list[str],
    metadatos_base: dict,
    cliente: OpenAI,
    batch_size: int = 50
) -> int:
    """Indexa chunks en la colección Chroma con embeddings."""
    total = len(chunks)
    for inicio in range(0, total, batch_size):
        lote       = chunks[inicio:min(inicio + batch_size, total)]
        embeddings = embeddear_batch(lote, cliente)
        ids        = [f"{metadatos_base['fuente_id']}_{i}" for i in range(inicio, inicio + len(lote))]
        metas      = [{
            **metadatos_base,
            "chunk_index":  i,
            "total_chunks": total,
            "n_chars":      len(c)
        } for i, c in enumerate(lote, inicio)]
        coleccion.upsert(ids=ids, documents=lote, embeddings=embeddings, metadatas=metas)
    return total


# ── Pipeline completo ─────────────────────────────────────────────────────────

def ingestar_fuentes(
    fuentes: list[dict],
    coleccion: chromadb.Collection,
    cliente: OpenAI
) -> dict:
    """
    Orquesta el pipeline completo para una lista de fuentes.

    Cada fuente es un dict con campos:
      - tipo:  "wikipedia" | "markdown" | "txt"
      - valor: título (Wikipedia) o ruta (archivo)
      - titulo: nombre descriptivo para metadatos
      - categoria: etiqueta de dominio (ej. "agri-food", "biotech")

    Devuelve estadísticas de ingesta.
    """
    stats = {"fuentes_ok": 0, "fuentes_error": 0, "chunks_totales": 0}
    fecha = datetime.date.today().isoformat()

    for fuente in fuentes:
        tipo     = fuente["tipo"]
        titulo   = fuente.get("titulo", fuente.get("valor", "sin_titulo"))
        print(f"  Ingesta [{tipo}]: '{titulo}'")

        try:
            if tipo == "wikipedia":
                texto = cargar_wikipedia(fuente["valor"], fuente.get("idioma", "en"))
            elif tipo == "markdown":
                texto = cargar_markdown(fuente["valor"])
            elif tipo == "txt":
                texto = cargar_txt(fuente["valor"])
            else:
                print(f"    Tipo desconocido: {tipo}")
                stats["fuentes_error"] += 1
                continue

            if not texto.strip():
                print(f"    Sin contenido.")
                stats["fuentes_error"] += 1
                continue

            texto  = limpiar(texto)
            chunks = chunkear(texto)
            n      = indexar(
                coleccion, chunks,
                metadatos_base={
                    "fuente":      titulo,
                    "fuente_id":   re.sub(r"\W+", "_", titulo.lower())[:30],
                    "tipo":        tipo,
                    "categoria":   fuente.get("categoria", "general"),
                    "fecha_ingesta": fecha,
                },
                cliente=cliente
            )
            print(f"    → {n} chunks indexados")
            stats["fuentes_ok"]      += 1
            stats["chunks_totales"]  += n

        except Exception as e:
            print(f"    ERROR: {e}")
            stats["fuentes_error"] += 1

    return stats
