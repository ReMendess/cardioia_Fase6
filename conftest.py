"""Configuração de testes na raiz do projeto.

Garante que o pacote ``cardioia`` (em ``src/``) seja importável quando a suíte roda a
partir da raiz do repositório, sem exigir caminho absoluto, instalação do pacote ou
variável de ambiente.
"""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ_DO_PROJETO = Path(__file__).resolve().parent
PASTA_SRC = RAIZ_DO_PROJETO / "src"

if str(PASTA_SRC) not in sys.path:
    sys.path.insert(0, str(PASTA_SRC))