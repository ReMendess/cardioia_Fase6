"""Agentes do CardioIA (Fase 6) — sistema multiagente com OpenAI Agents SDK.

Os quatro agentes da atividade, com handoffs **reais** do SDK:

- :mod:`~cardioia.agents.risk_analyst` — Analista de Risco (tool do modelo de ML);
- :mod:`~cardioia.agents.protocol_specialist` — Especialista em Protocolos (tool determinística);
- :mod:`~cardioia.agents.response_planner` — Planejador de Resposta (saída estruturada);
- :mod:`~cardioia.agents.orchestrator` — Orquestrador (coordena e monta o contrato).

Apoio: :mod:`~cardioia.agents.contexto` (contexto tipado + trilha) e
:mod:`~cardioia.agents.modelo_simulado` (execução sem API, para testes e demonstração).

Execução típica:

    from cardioia.agents import orchestrator

    resposta = orchestrator.executar_pipeline(paciente)     # modo simulado por padrão
    print(resposta.resumo_textual())
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "contexto",
    "modelo_simulado",
    "orchestrator",
    "protocol_specialist",
    "response_planner",
    "risk_analyst",
    "executar_pipeline",
]

_SUBMODULOS = (
    "contexto",
    "modelo_simulado",
    "protocol_specialist",
    "response_planner",
    "risk_analyst",
    "orchestrator",
)
_FUNCOES = {"executar_pipeline": "orchestrator"}


def __getattr__(nome: str) -> Any:
    """Importa submódulos sob demanda (PEP 562), como em ``cardioia.ml``."""
    if nome in _SUBMODULOS:
        modulo = import_module(f".{nome}", __name__)
        globals()[nome] = modulo
        return modulo
    if nome in _FUNCOES:
        valor = getattr(import_module(f".{_FUNCOES[nome]}", __name__), nome)
        globals()[nome] = valor
        return valor
    raise AttributeError(f"module {__name__!r} has no attribute {nome!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})