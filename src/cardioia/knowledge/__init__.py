"""Camada de conhecimento do CardioIA (Fase 6).

Expõe a base de protocolos oficial de forma **estruturada e determinística**:

    from cardioia.knowledge import protocols

    protocolos = protocols.carregar_protocolos()
    avaliacao = protocols.avaliar_protocolos(paciente, pico_risco_previsto=1)
    print(avaliacao.resumo())

Nenhuma decisão desta camada passa por LLM: as regras são aplicadas em Python, com os
operadores e limiares exatos do material oficial.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["protocols"]

_SUBMODULOS = ("protocols",)


def __getattr__(nome: str) -> Any:
    """Importa submódulos sob demanda (PEP 562), como em ``cardioia.ml``."""
    if nome in _SUBMODULOS:
        modulo = import_module(f".{nome}", __name__)
        globals()[nome] = modulo
        return modulo
    raise AttributeError(f"module {__name__!r} has no attribute {nome!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})