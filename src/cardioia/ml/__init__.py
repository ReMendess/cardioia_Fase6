"""Camada de Machine Learning do CardioIA (Fase 6).

Expõe as constantes de domínio (autoridade única sobre a base e o rótulo oficial) e
os módulos de trabalho:

- ``dataset``  : carrega/valida a base oficial e prepara ``X``/``y``;
- ``evaluate`` : métricas, matriz de confusão e relatório serializáveis em JSON;
- ``train``    : treina, compara pelo critério pré-declarado e salva os artefatos;
- ``predict``  : carrega o modelo salvo e prevê ``pico_risco`` para novos casos.

Uso:

    from cardioia.ml import dataset, evaluate, predict, train

    df = dataset.carregar_base_oficial()
    X, y = dataset.separar_features_e_alvo(df)
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "dataset",
    "evaluate",
    "predict",
    "train",
    "VARIAVEL_ALVO",
    "COLUNAS_PREDITORAS",
    "COLUNAS_AUXILIARES_IGNORADAS",
    "SEMENTE",
    "TAMANHO_TESTE",
    "CORTE_RISK_SCORE",
    "LIMIARES_DO_ROTULO",
]

_SUBMODULOS = ("dataset", "evaluate", "predict", "train")
_CONSTANTES = (
    "VARIAVEL_ALVO",
    "COLUNAS_PREDITORAS",
    "COLUNAS_AUXILIARES_IGNORADAS",
    "SEMENTE",
    "TAMANHO_TESTE",
    "CORTE_RISK_SCORE",
    "LIMIARES_DO_ROTULO",
)


def __getattr__(nome: str) -> Any:
    """Importa submódulos e constantes sob demanda (PEP 562).

    A carga é preguiçosa de propósito: ``python -m src.cardioia.ml.train`` executa este
    ``__init__`` antes de rodar o módulo de treino, e a importação ansiosa do próprio
    módulo alvo faria o ``runpy`` avisar sobre comportamento imprevisível.
    """
    if nome in _SUBMODULOS:
        modulo = import_module(f".{nome}", __name__)
        globals()[nome] = modulo
        return modulo
    if nome in _CONSTANTES:
        valor = getattr(import_module(".dataset", __name__), nome)
        globals()[nome] = valor
        return valor
    raise AttributeError(f"module {__name__!r} has no attribute {nome!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})