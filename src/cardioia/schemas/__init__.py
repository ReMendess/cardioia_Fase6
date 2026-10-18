"""Contratos Pydantic do CardioIA (Fase 6).

    from cardioia.schemas import risk_result

    resposta = risk_result.RespostaDaExecucao(...)
    resposta.como_dicionario()   # JSON-safe

O contrato único de saída do pipeline é ``RespostaDaExecucao``: usado pelo orquestrador,
pela CLI e pelos testes, sem estruturas equivalentes duplicadas.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["risk_result"]

_SUBMODULOS = ("risk_result",)


def __getattr__(nome: str) -> Any:
    """Importa submódulos sob demanda (PEP 562), como em ``cardioia.ml`` e ``knowledge``."""
    if nome in _SUBMODULOS:
        modulo = import_module(f".{nome}", __name__)
        globals()[nome] = modulo
        return modulo
    raise AttributeError(f"module {__name__!r} has no attribute {nome!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})