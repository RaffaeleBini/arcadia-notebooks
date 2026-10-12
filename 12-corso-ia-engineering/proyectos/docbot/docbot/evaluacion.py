"""
evaluacion.py — Evaluador de faithfulness y métricas de calidad del pipeline RAG.
"""

import json
import statistics
from dataclasses import dataclass, field
from openai import OpenAI

MODELO = "gpt-4o-mini"

SYSTEM_FAITHFULNESS = """You are a strict fact-checker. Evaluate whether each factual claim
in the answer is supported by the provided context. Be precise."""

PROMPT_FAITHFULNESS = """CONTEXT:
{contexto}

QUESTION: {pregunta}
ANSWER: {respuesta}

For each concrete factual claim in the answer, check if it is supported by the context.
Return JSON:
{{
  "claims": [{{"claim": "...", "supported": true/false}}],
  "faithfulness_score": 0.0-1.0,
  "verdict": "FAITHFUL" | "PARTIALLY_FAITHFUL" | "UNFAITHFUL"
}}"""


@dataclass
class ResultadoEval:
    pregunta:          str
    respondida:        bool
    faithfulness:      float | None
    context_relevance: float
    verdict:           str
    claims_no_respaldadas: list[str] = field(default_factory=list)


def evaluar_turno(
    pregunta: str,
    respuesta: str,
    chunks_textos: list[str],
    chunks_scores: list[float],
    cliente: OpenAI
) -> ResultadoEval:
    """Evalúa un único turno de conversación."""
    respondida        = "don't have enough information" not in respuesta.lower()
    context_relevance = round(statistics.mean(chunks_scores), 4) if chunks_scores else 0.0

    if not respondida or not chunks_textos:
        return ResultadoEval(
            pregunta=pregunta, respondida=respondida,
            faithfulness=None, context_relevance=context_relevance,
            verdict="NOT_ANSWERED"
        )

    contexto = "\n\n---\n\n".join(chunks_textos)[:3000]
    prompt   = PROMPT_FAITHFULNESS.format(
        contexto=contexto, pregunta=pregunta, respuesta=respuesta[:500]
    )

    try:
        r     = cliente.chat.completions.create(
            model=MODELO,
            messages=[
                {"role": "system", "content": SYSTEM_FAITHFULNESS},
                {"role": "user",   "content": prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        datos = json.loads(r.choices[0].message.content)
        no_respaldadas = [c["claim"] for c in datos.get("claims", []) if not c.get("supported", True)]
        return ResultadoEval(
            pregunta=pregunta, respondida=respondida,
            faithfulness=datos.get("faithfulness_score"),
            context_relevance=context_relevance,
            verdict=datos.get("verdict", "UNKNOWN"),
            claims_no_respaldadas=no_respaldadas
        )
    except Exception as e:
        return ResultadoEval(
            pregunta=pregunta, respondida=respondida,
            faithfulness=None, context_relevance=context_relevance,
            verdict=f"EVAL_ERROR: {e}"
        )


def resumen_evaluaciones(evaluaciones: list[ResultadoEval]) -> dict:
    """Calcula métricas agregadas de un conjunto de evaluaciones."""
    respondidas = [e for e in evaluaciones if e.respondida]
    scores_faith = [e.faithfulness for e in respondidas if e.faithfulness is not None]
    scores_ctx   = [e.context_relevance for e in respondidas]

    return {
        "n_total":              len(evaluaciones),
        "n_respondidas":        len(respondidas),
        "tasa_respuesta":       round(len(respondidas) / max(len(evaluaciones), 1), 3),
        "faithfulness_media":   round(statistics.mean(scores_faith), 3) if scores_faith else None,
        "context_rel_media":    round(statistics.mean(scores_ctx), 3)   if scores_ctx   else None,
        "n_con_alucinaciones":  sum(1 for e in respondidas if e.faithfulness is not None and e.faithfulness < 0.70),
        "distribucion_veredictos": {
            v: sum(1 for e in evaluaciones if e.verdict == v)
            for v in {"FAITHFUL", "PARTIALLY_FAITHFUL", "UNFAITHFUL", "NOT_ANSWERED", "EVAL_ERROR"}
            if any(e.verdict == v for e in evaluaciones)
        }
    }
