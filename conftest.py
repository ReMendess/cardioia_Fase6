"""Configuração de testes na raiz do projeto.

Garante que o pacote ``cardioia`` (em ``src/``) seja importável quando a suíte roda a
partir da raiz do repositório, sem exigir caminho absoluto, instalação do pacote ou
variável de ambiente.

**Ordem de busca (``sys.path.append``, não ``insert``)**: ``src/`` vai para o *fim* da
lista de propósito. O repositório possui ``src/agents/`` (placeholders da Etapa 1) e o
OpenAI Agents SDK instala um pacote de topo chamado ``agents``; se ``src/`` viesse
primeiro, ``import agents`` resolveria para a pasta vazia do repositório e o SDK ficaria
inacessível. Ver ``tests/test_agents.py::test_sdk_nao_e_sombreado``.
"""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ_DO_PROJETO = Path(__file__).resolve().parent
PASTA_SRC = RAIZ_DO_PROJETO / "src"

if str(PASTA_SRC) not in sys.path:
    sys.path.append(str(PASTA_SRC))