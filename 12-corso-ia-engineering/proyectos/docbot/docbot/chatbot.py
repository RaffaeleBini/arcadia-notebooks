"""
chatbot.py — Clase principal DocBot. Orquesta RAG + memoria + evaluación.
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

from .ingesta    import ingestar_fuentes
from .retrieval  import pipeline_consulta, ResultadoRetrieval
from .memoria    import (Sesion, extraer_hechos, guardar_hechos,
                         recuperar_hechos, borrar_memoria_semantica,
                         borrar_todas_de)
from .evaluacion import evaluar_turno, resumen_evaluaciones, ResultadoEval

load_dotenv()

SYSTEM_BASE = """You are DocBot, a documentation assistant with memory.
You answer questions based strictly on indexed documents.
You remember the user across sessions.
When the user mentions personal details, acknowledge them naturally."""


@dataclass
class RespuestaDocBot:
    respuesta:    str
    fuentes:      list[str]
    n_chunks:     int
    respondida:   bool
    latencia_ms:  float
    faithfulness: float | None = None


class DocBot:
    """
    Chatbot RAG con memoria de sesión y semántica.

    Uso básico:
        bot = DocBot(user_id="usuario1")
        bot.ingestar(fuentes=[{"tipo": "wikipedia", "valor": "FastAPI", "titulo": "FastAPI"}])
        r = bot.responder("What is FastAPI?")
        print(r.respuesta)
    """

    def __init__(
        self,
        user_id:        str,
        coleccion_docs: str  = "rag_docs",
        evaluar_online: bool = False,
        nueva_sesion:   bool = False,
    ):
        self.user_id        = user_id
        self.evaluar_online = evaluar_online

        self._cliente = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._chroma  = chromadb.PersistentClient(path="./chroma_db")

        # Colección de documentos
        try:
            self._coleccion = self._chroma.get_collection(coleccion_docs)
        except Exception:
            self._coleccion = self._chroma.create_collection(
                coleccion_docs, metadata={"hnsw:space": "cosine"}
            )

        # Sesión: carga la más reciente o crea una nueva
        self._sesion = (
            Sesion.nueva(user_id, SYSTEM_BASE)
            if nueva_sesion
            else (Sesion.cargar_ultima(user_id) or Sesion.nueva(user_id, SYSTEM_BASE))
        )

        # Buffer de evaluaciones de la sesión actual
        self._evaluaciones: list[ResultadoEval] = []

    # ── API pública ───────────────────────────────────────────────────────────

    def responder(self, pregunta: str) -> RespuestaDocBot:
        """
        Pipeline completo:
        1. Enriquece el system prompt con memoria semántica del usuario
        2. Corre el pipeline RAG con historial de conversación
        3. Guarda el turno en la sesión
        4. Comprime el historial si es necesario
        5. Evalúa la respuesta (si evaluar_online=True)
        """
        # 1. Recupera hechos del usuario relevantes para esta pregunta
        hechos = recuperar_hechos(pregunta, self.user_id, self._chroma, self._cliente)
        contexto_usuario = ""
        if hechos:
            contexto_usuario = "\n\nUSER CONTEXT (from memory):\n" + "\n".join(f"- {h}" for h in hechos)

        # 2. Pipeline RAG
        resultado: ResultadoRetrieval = pipeline_consulta(
            pregunta=pregunta,
            coleccion=self._coleccion,
            cliente=self._cliente,
            historial=self._sesion.historial_sin_system(),
            k=4,
            usar_rewriting=True
        )

        # 3. Si hay contexto de usuario, inyéctalo en la respuesta si no fue respondida
        if hechos and not resultado.respondida:
            # Intento secundario: el LLM puede responder con conocimiento previo del usuario
            # pero sin inventar hechos del documento
            pass

        # 4. Guarda el turno
        self._sesion.agregar("user",      pregunta)
        self._sesion.agregar("assistant", resultado.respuesta)
        self._sesion.comprimir_si_necesario(self._cliente)
        self._sesion.guardar()

        # 5. Evaluación online (opcional, tiene coste)
        faithfulness = None
        if self.evaluar_online and resultado.respondida:
            eval_r = evaluar_turno(
                pregunta=pregunta,
                respuesta=resultado.respuesta,
                chunks_textos=[c.texto for c in resultado.chunks],
                chunks_scores=[c.similitud for c in resultado.chunks],
                cliente=self._cliente
            )
            self._evaluaciones.append(eval_r)
            faithfulness = eval_r.faithfulness

        fuentes_unicas = list(dict.fromkeys(c.fuente for c in resultado.chunks))

        return RespuestaDocBot(
            respuesta=resultado.respuesta,
            fuentes=fuentes_unicas,
            n_chunks=len(resultado.chunks),
            respondida=resultado.respondida,
            latencia_ms=resultado.latencia_ms,
            faithfulness=faithfulness
        )

    def ingestar(self, fuentes: list[dict]) -> dict:
        """Indexa nuevos documentos en la colección. Devuelve estadísticas."""
        print(f"Iniciando ingesta de {len(fuentes)} fuente(s)...")
        stats = ingestar_fuentes(fuentes, self._coleccion, self._cliente)
        print(f"Ingesta completada: {stats}")
        return stats

    def evaluar_sesion(self, n_ultimos: int = 10) -> dict:
        """
        Evalúa los últimos n_ultimos turnos de la sesión.
        Más lento pero más completo que la evaluación online.
        """
        historial = self._sesion.historial_sin_system()
        # Reconstruye pares pregunta/respuesta
        pares = []
        for i in range(0, len(historial) - 1, 2):
            if historial[i]["role"] == "user" and historial[i+1]["role"] == "assistant":
                pares.append((historial[i]["content"], historial[i+1]["content"]))

        pares_a_evaluar = pares[-n_ultimos:]
        evaluaciones    = []

        for pregunta, respuesta in pares_a_evaluar:
            # Re-recupera los chunks para esta pregunta
            from .retrieval import recuperar, generar_embedding
            chunks = recuperar(pregunta, self._coleccion, self._cliente, k=4)
            chunks_relevantes = [c for c in chunks if c.similitud >= 0.65]

            eval_r = evaluar_turno(
                pregunta=pregunta,
                respuesta=respuesta,
                chunks_textos=[c.texto for c in chunks_relevantes],
                chunks_scores=[c.similitud for c in chunks_relevantes],
                cliente=self._cliente
            )
            evaluaciones.append(eval_r)

        return resumen_evaluaciones(evaluaciones)

    def cerrar_sesion(self) -> None:
        """Al cerrar: extrae hechos del usuario y los guarda en memoria semántica."""
        hechos = extraer_hechos(self._sesion, self._cliente)
        n      = guardar_hechos(hechos, self.user_id, self._sesion.session_id,
                                self._chroma, self._cliente)
        self._sesion.guardar()
        print(f"Sesión cerrada. {n} hechos guardados en memoria semántica.")

    def resetear(self) -> None:
        """Borra solo la sesión actual y comienza una nueva."""
        self._sesion.borrar()
        self._sesion = Sesion.nueva(self.user_id, SYSTEM_BASE)
        self._evaluaciones = []

    def olvidar(self) -> None:
        """Borra TODA la memoria del usuario (sesiones + memoria semántica)."""
        n_sesiones = borrar_todas_de(self.user_id)
        borrar_memoria_semantica(self.user_id, self._chroma)
        self._sesion    = Sesion.nueva(self.user_id, SYSTEM_BASE)
        self._evaluaciones = []
        print(f"Memoria borrada: {n_sesiones} sesiones + memoria semántica.")

    @property
    def n_docs(self) -> int:
        """Número de chunks indexados en la colección."""
        return self._coleccion.count()

    @property
    def session_id(self) -> str:
        return self._sesion.session_id
