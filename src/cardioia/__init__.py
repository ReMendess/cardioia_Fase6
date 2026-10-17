"""Pacote principal do CardioIA (Fase 6).

Contém a camada reutilizável de Machine Learning (``cardioia.ml``) que o notebook
``notebooks/treinamento_modelo.ipynb``, os testes automatizados e, na próxima etapa,
os agentes do OpenAI Agents SDK consomem.

Como importar (sem depender de caminho absoluto da máquina):

    import sys
    from pathlib import Path

    raiz = Path.cwd()                     # execute a partir da raiz do repositório
    sys.path.insert(0, str(raiz / "src"))

    from cardioia.ml import dataset, evaluate, predict, train
"""

from __future__ import annotations

__all__ = ["ml"]
__versao__ = "0.2.0"