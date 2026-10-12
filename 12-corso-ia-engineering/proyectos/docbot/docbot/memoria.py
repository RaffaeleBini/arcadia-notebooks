"""
memoria.py — Gestión de sesiones persistidas y memoria semántica por usuario.
"""

import json
import uuid
import hashlib
import datetime
from dataclasses import dataclass, field, asdict
from pathlib import Path
from openai import OpenAI
import chromadb

MODELO_EMBEDDING = "text-embedding-3-small"
MODELO           = "gpt-4o-mini"
DIR_SESIONES     = Path("sesiones")
DIR_SESIONES.mkdir(exist_ok=True)


# ── Sesión ────────────────────────────────────────────────────────────────────

@dataclass
class Sesion:
    session_id:  str
    user_id:     str
    created_at:  str
    updated_at:  str
    mensajes:    list[dict] = field(default_factory=list)
    n_turnos:    int        = 0

    @classmethod
    def nueva(cls, user_id: str, system_prompt: str | None = None) -> "Sesion":
        ahora = datetime.datetime.utcnow().isoformat()
        s     = cls(
            session_id=f"sess_{uuid.uuid4().hex[:8]}",
            user_id=user_id, created_at=ahora, updated_at=ahora
        )
        if system_prompt:
            s.mensajes.append({"role": "system", "content": system_prompt})
        return s

    def agregar(self, rol: str, contenido: str) -> None:
        self.mensajes.append({"role": rol, "content": contenido})
        self.updated_at = datetime.datetime.utcnow().isoformat()
        self.n_turnos   = sum(1 for m in self.mensajes if m["role"] == "user")

    def guardar(self) -> None:
        ruta = DIR_SESIONES / f"{self.session_id}.json"
        with open(ruta, "w") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    @classmethod
    def cargar_ultima(cls, user_id: str) -> "Sesion | None":
        """Carga la sesión más reciente del usuario."""
        sesiones = sorted(
            DIR_SESIONES.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        for ruta in sesiones:
            with open(ruta) as f:
                datos = json.load(f)
            if datos.get("user_id") == user_id:
                return cls(**datos)
        return None

    def historial_sin_system(self) -> list[dict]:
        return [m for m in self.mensajes if m["role"] != "system"]

    def comprimir_si_necesario(self, cliente: OpenAI, umbral: int = 14, conservar: int = 6) -> None:
        """Comprime el historial cuando supera el umbral."""
        resto = self.historial_sin_system()
        if len(resto) <= umbral:
            return
        a_comprimir = resto[:-conservar]
        a_conservar = resto[-conservar:]
        texto = "\n".join(f"{m['role'].upper()}: {m['content'][:200]}" for m in a_comprimir)
        resumen = cliente.chat.completions.create(
            model=MODELO,
            messages=[
                {"role": "system",  "content": "Summarize this conversation in 2-3 sentences, preserving key facts about the user."},
                {"role": "user",    "content": texto}
            ],
            temperature=0.0, max_tokens=150
        ).choices[0].message.content

        sistema  = [m for m in self.mensajes if m["role"] == "system"]
        self.mensajes = sistema + [
            {"role": "system", "content": f"[Earlier conversation summary]: {resumen}"}
        ] + a_conservar

    def borrar(self) -> None:
        ruta = DIR_SESIONES / f"{self.session_id}.json"
        if ruta.exists():
            ruta.unlink()

    @staticmethod
    def borrar_todas_de(user_id: str) -> int:
        n = 0
        for ruta in DIR_SESIONES.glob("*.json"):
            with open(ruta) as f:
                datos = json.load(f)
            if datos.get("user_id") == user_id:
                ruta.unlink()
                n += 1
        return n


# ── Memoria semántica ─────────────────────────────────────────────────────────

SYSTEM_EXTRACTOR = """Extract durable facts about the user from this conversation.
Only include: name, role, company, location, preferences, goals, domain knowledge.
Skip conversational filler. Return JSON: {"hechos": [{"texto": "...", "categoria": "perfil|preferencia|objetivo|conocimiento"}]}"""


def extraer_hechos(sesion: Sesion, cliente: OpenAI) -> list[dict]:
    texto = "\n".join(
        f"{m['role'].upper()}: {m['content'][:200]}"
        for m in sesion.mensajes if m["role"] in ("user", "assistant")
    )
    r = cliente.chat.completions.create(
        model=MODELO,
        messages=[
            {"role": "system", "content": SYSTEM_EXTRACTOR},
            {"role": "user",   "content": f"Conversation:\n{texto[:3000]}"}
        ],
        temperature=0.0, max_tokens=400,
        response_format={"type": "json_object"}
    )
    datos = json.loads(r.choices[0].message.content)
    return datos.get("hechos", [])


def guardar_hechos(hechos: list[dict], user_id: str, session_id: str,
                   chroma: chromadb.Client, cliente: OpenAI) -> int:
    if not hechos:
        return 0
    nombre    = f"memoria_{user_id}"
    try:
        col   = chroma.get_collection(nombre)
    except Exception:
        col   = chroma.create_collection(nombre, metadata={"hnsw:space": "cosine"})

    textos    = [h["texto"] for h in hechos]
    ids       = ["fact_" + hashlib.md5(t.encode()).hexdigest()[:8] for t in textos]
    ahora     = datetime.datetime.utcnow().isoformat()
    r         = cliente.embeddings.create(model=MODELO_EMBEDDING, input=textos)
    embeddings = [item.embedding for item in sorted(r.data, key=lambda x: x.index)]
    metas     = [{"user_id": user_id, "categoria": h.get("categoria", "general"),
                  "session_id": session_id, "created_at": ahora} for h in hechos]
    col.upsert(ids=ids, documents=textos, embeddings=embeddings, metadatas=metas)
    return len(hechos)


def recuperar_hechos(pregunta: str, user_id: str,
                     chroma: chromadb.Client, cliente: OpenAI,
                     k: int = 4, umbral: float = 0.60) -> list[str]:
    nombre = f"memoria_{user_id}"
    try:
        col = chroma.get_collection(nombre)
    except Exception:
        return []
    if col.count() == 0:
        return []
    vector = cliente.embeddings.create(model=MODELO_EMBEDDING, input=pregunta).data[0].embedding
    resp   = col.query(
        query_embeddings=[vector],
        n_results=min(k, col.count()),
        include=["documents", "distances"],
        where={"user_id": user_id}
    )
    return [
        doc for doc, dist in zip(resp["documents"][0], resp["distances"][0])
        if (1 - dist) >= umbral
    ]


def borrar_memoria_semantica(user_id: str, chroma: chromadb.Client) -> bool:
    nombre = f"memoria_{user_id}"
    try:
        chroma.delete_collection(nombre)
        return True
    except Exception:
        return False
