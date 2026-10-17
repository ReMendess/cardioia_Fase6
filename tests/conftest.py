"""Fixtures compartilhadas da suíte de testes do CardioIA (Fase 6).

Regras seguidas pelas fixtures:

- os testes **não** acessam a internet nem a API da OpenAI;
- nenhum teste depende do ``.pkl`` versionado: quando é preciso um modelo treinado, a
  fixture :func:`modelo_temporario` treina um Random Forest pequeno (10 árvores, semente
  42) numa amostra de 400 linhas e grava os artefatos em ``tmp_path_factory``;
- a base oficial é carregada **uma única vez** por sessão (5.000 linhas, ~1 s).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

RAIZ_DO_PROJETO = Path(__file__).resolve().parents[1]
PASTA_SRC = RAIZ_DO_PROJETO / "src"
if str(PASTA_SRC) not in sys.path:
    sys.path.insert(0, str(PASTA_SRC))

from cardioia.ml import dataset, train  # noqa: E402 - depende do sys.path acima

LINHAS_DA_AMOSTRA = 400
ARVORES_DO_MODELO_DE_TESTE = 10


@pytest.fixture(scope="session")
def raiz_do_projeto() -> Path:
    """Raiz do repositório (pasta que contém ``data/base_dados.py``)."""
    raiz = dataset.localizar_raiz_do_projeto()
    assert raiz is not None, "Raiz do projeto nao encontrada a partir do diretorio de testes."
    return raiz


@pytest.fixture(scope="session")
def base_oficial(raiz_do_projeto: Path):
    """Base sintética do material oficial (5.000 x 11), carregada uma vez por sessão."""
    return dataset.carregar_base_oficial(raiz=raiz_do_projeto)


@pytest.fixture(scope="session")
def amostra_pequena(base_oficial):
    """Amostra determinística de 400 linhas, usada para treinar modelos de teste rápido."""
    return base_oficial.sample(n=LINHAS_DA_AMOSTRA, random_state=dataset.SEMENTE).reset_index(drop=True)


@pytest.fixture(scope="session")
def paciente_da_aula(raiz_do_projeto: Path) -> dict[str, float]:
    """Caso de simulação lido de ``data/novo_paciente.py`` (coluna já normalizada)."""
    return dataset.carregar_caso_do_paciente_da_aula(raiz=raiz_do_projeto)


@pytest.fixture(scope="session")
def modelo_temporario(tmp_path_factory, amostra_pequena) -> dict:
    """Treina e salva um modelo pequeno em diretório temporário (sem tocar em ``models/``).

    Returns:
        Dicionário com ``pasta``, ``caminho_modelo``, ``caminho_metadados``, ``metadados``,
        ``resultado`` (retorno de ``treinar_e_avaliar``) e ``modelo``.
    """
    pasta = tmp_path_factory.mktemp("modelos_cardioia")
    resultado = train.treinar_e_avaliar(
        df=amostra_pequena,
        n_estimators=ARVORES_DO_MODELO_DE_TESTE,
        origem_da_base="fixture de teste (amostra da base oficial)",
    )
    artefatos = train.salvar_artefatos(resultado, pasta_modelos=pasta)
    return {
        "pasta": pasta,
        "caminho_modelo": artefatos["caminho_modelo"],
        "caminho_metadados": artefatos["caminho_metadados"],
        "metadados": artefatos["metadados"],
        "resultado": resultado,
        "modelo": resultado["modelo_escolhido"],
        "avaliacoes": resultado["avaliacoes"],
        "comparativo": resultado["comparativo"],
    }