"""Testes da camada de dados (``src/cardioia/ml/dataset.py``).

Cobrem: carregamento do material oficial, dimensões e colunas, presença e contrato da
variável-alvo, ausência de ``risk_score`` entre as features, detecção de valores ausentes
e entradas inválidas, reprodutibilidade da geração e leitura do caso do paciente da aula.

Nenhum teste acessa a internet ou a API da OpenAI.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cardioia.ml import dataset

COLUNAS_ESPERADAS = [*dataset.COLUNAS_PREDITORAS, dataset.VARIAVEL_ALVO]

# --------------------------------------------------------------------------- #
# Carregamento do material oficial
# --------------------------------------------------------------------------- #


def test_carregar_base_oficial_tem_dimensoes_esperadas(base_oficial):
    assert base_oficial.shape == (5000, 11)


def test_colunas_esperadas_e_na_ordem_do_material(base_oficial):
    assert list(base_oficial.columns) == COLUNAS_ESPERADAS


def test_variavel_alvo_presente_e_binaria(base_oficial):
    assert dataset.VARIAVEL_ALVO in base_oficial.columns
    assert sorted(base_oficial[dataset.VARIAVEL_ALVO].unique().tolist()) == [0, 1]


def test_base_oficial_nao_tem_valores_ausentes(base_oficial):
    assert int(base_oficial.isna().sum().sum()) == 0


def test_origem_da_base_aponta_para_o_arquivo_oficial(raiz_do_projeto):
    _, informacoes = dataset.carregar_base_oficial(raiz=raiz_do_projeto, retornar_origem=True)
    assert "base_dados.py" in informacoes["origem"]
    assert informacoes["usado_failover"] is False


def test_reprodutibilidade_da_geracao(base_oficial, raiz_do_projeto):
    segunda = dataset.carregar_base_oficial(raiz=raiz_do_projeto)
    pd.testing.assert_frame_equal(base_oficial, segunda)
    assert dataset.sha256_da_base(segunda) == dataset.sha256_da_base(base_oficial)


def test_sha256_da_base_confere_com_o_metadata_versionado(base_oficial, raiz_do_projeto):
    caminho = raiz_do_projeto / "models" / "modelo_pico_risco_metadata.json"
    if not caminho.is_file():
        pytest.skip("metadata do modelo ainda nao gerado neste repositorio")
    registrado = json.loads(caminho.read_text(encoding="utf-8"))["sha256_da_base_usada"]
    assert dataset.sha256_da_base(base_oficial) == registrado


def test_carregamento_por_codigo_fonte_explicito(raiz_do_projeto):
    codigo = dataset.ler_arquivo_oficial(dataset.NOME_ARQUIVO_BASE, raiz=raiz_do_projeto)
    df, informacoes = dataset.carregar_base_oficial(codigo_fonte=codigo, retornar_origem=True)
    assert df.shape == (5000, 11)
    assert informacoes["usado_failover"] is True
    esperado = dataset.carregar_base_oficial(raiz=raiz_do_projeto)
    assert dataset.sha256_da_base(df) == dataset.sha256_da_base(esperado)


def test_arquivo_oficial_ausente_gera_erro_claro(tmp_path):
    with pytest.raises(dataset.ArquivoOficialNaoEncontradoError) as erro:
        dataset.carregar_base_oficial(raiz=tmp_path)
    mensagem = str(erro.value)
    assert "base_dados.py" in mensagem
    assert "data/" in mensagem


def test_ler_arquivo_oficial_inexistente_gera_erro(tmp_path):
    with pytest.raises(dataset.ArquivoOficialNaoEncontradoError):
        dataset.ler_arquivo_oficial("nao_existe.py", raiz=tmp_path)


def test_localizar_raiz_do_projeto_encontra_o_repositorio(raiz_do_projeto):
    assert (raiz_do_projeto / "data" / "base_dados.py").is_file()
    assert dataset.localizar_raiz_do_projeto(raiz_do_projeto / "tests") == raiz_do_projeto


def test_localizar_raiz_do_projeto_retorna_none_sem_marcador(tmp_path):
    assert dataset.localizar_raiz_do_projeto(tmp_path) is None


# --------------------------------------------------------------------------- #
# Validação
# --------------------------------------------------------------------------- #


def test_validar_base_retorna_resumo_completo(base_oficial):
    resumo = dataset.validar_base(base_oficial)
    assert resumo["dimensoes"] == {"linhas": 5000, "colunas": 11}
    assert resumo["valores_ausentes_total"] == 0
    assert resumo["linhas_duplicadas"] == 0
    assert resumo["alvo_valores"] == [0, 1]
    assert 0.25 < resumo["alvo_proporcao_positiva"] < 0.28


def test_validar_base_detecta_coluna_ausente(base_oficial):
    incompleta = base_oficial.drop(columns=["spo2_media"])
    with pytest.raises(dataset.BaseInvalidaError) as erro:
        dataset.validar_base(incompleta)
    assert "spo2_media" in str(erro.value)


def test_validar_base_detecta_valores_ausentes(base_oficial):
    com_nulo = base_oficial.copy()
    com_nulo.loc[0, "fc_media"] = np.nan
    with pytest.raises(dataset.BaseInvalidaError) as erro:
        dataset.validar_base(com_nulo)
    assert "ausentes" in str(erro.value)


def test_validar_base_detecta_tipo_nao_numerico(base_oficial):
    com_texto = base_oficial.copy()
    com_texto["temperatura"] = com_texto["temperatura"].astype(str)
    with pytest.raises(dataset.BaseInvalidaError) as erro:
        dataset.validar_base(com_texto)
    assert "numericas" in str(erro.value)


def test_validar_base_detecta_alvo_nao_binario(base_oficial):
    alvo_invalido = base_oficial.copy()
    alvo_invalido.loc[0, dataset.VARIAVEL_ALVO] = 2
    with pytest.raises(dataset.BaseInvalidaError) as erro:
        dataset.validar_base(alvo_invalido)
    assert "binaria" in str(erro.value)


def test_validar_base_detecta_base_vazia():
    with pytest.raises(dataset.BaseInvalidaError):
        dataset.validar_base(pd.DataFrame())


def test_validar_base_detecta_entrada_de_tipo_errado():
    with pytest.raises(dataset.BaseInvalidaError):
        dataset.validar_base([1, 2, 3])  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Features, alvo e ausência de vazamento
# --------------------------------------------------------------------------- #


def test_separar_features_e_alvo_devolve_X_e_y(base_oficial):
    X, y = dataset.separar_features_e_alvo(base_oficial)
    assert list(X.columns) == list(dataset.COLUNAS_PREDITORAS)
    assert y.name == dataset.VARIAVEL_ALVO
    assert len(X) == len(y) == len(base_oficial)


def test_risk_score_nunca_entra_nas_features(base_oficial):
    X, _ = dataset.separar_features_e_alvo(base_oficial)
    assert "risk_score" not in X.columns
    assert not any("risk" in coluna for coluna in X.columns)


def test_coluna_auxiliar_risk_score_e_ignorada(base_oficial):
    com_auxiliar = base_oficial.copy()
    com_auxiliar["risk_score"] = dataset.calcular_risk_score(com_auxiliar)
    X, y = dataset.separar_features_e_alvo(com_auxiliar)
    assert "risk_score" not in X.columns
    assert list(X.columns) == list(dataset.COLUNAS_PREDITORAS)
    assert y.equals(com_auxiliar[dataset.VARIAVEL_ALVO])


def test_separar_features_e_alvo_erro_quando_alvo_ausente(base_oficial):
    sem_alvo = base_oficial.drop(columns=[dataset.VARIAVEL_ALVO])
    with pytest.raises(dataset.BaseInvalidaError):
        dataset.separar_features_e_alvo(sem_alvo)


# --------------------------------------------------------------------------- #
# Regra determinística do material oficial
# --------------------------------------------------------------------------- #


def test_regra_deterministica_reproduz_o_rotulo(base_oficial):
    recalculado = dataset.rotulo_pela_regra(base_oficial)
    assert float((recalculado == base_oficial[dataset.VARIAVEL_ALVO]).mean()) == 1.0
    assert recalculado.equals(base_oficial[dataset.VARIAVEL_ALVO])


def test_calcular_risk_score_fica_entre_0_e_6(base_oficial):
    risk_score = dataset.calcular_risk_score(base_oficial)
    assert int(risk_score.min()) >= 0
    assert int(risk_score.max()) <= 6
    assert risk_score.dtype.kind in "iu"


def test_indicadores_ativos_do_paciente_da_aula(paciente_da_aula):
    ativos = dataset.indicadores_ativos(paciente_da_aula)
    assert len(ativos) == 6  # todos os indicadores do rótulo disparam para este caso
    assert "disponibilidade_recursos < 40" in ativos


def test_caso_do_paciente_da_aula_e_normalizado(paciente_da_aula):
    assert set(paciente_da_aula) == set(dataset.COLUNAS_PREDITORAS)
    assert "disponibilidade recursos" not in paciente_da_aula
    assert paciente_da_aula["disponibilidade_recursos"] == 25.0
    assert paciente_da_aula["carga_sistema"] == 92.0
    assert paciente_da_aula["idade_media"] == 62.0


def test_normalizar_caso_de_paciente_corrige_coluna_com_espaco():
    corrigido = dataset.normalizar_caso_de_paciente({"disponibilidade recursos": 25})
    assert corrigido == {"disponibilidade_recursos": 25.0}


# --------------------------------------------------------------------------- #
# Divisão treino/teste e faixas
# --------------------------------------------------------------------------- #


def test_divisao_treino_teste_e_estratificada_e_disjunta(base_oficial):
    X, y = dataset.separar_features_e_alvo(base_oficial)
    X_treino, X_teste, y_treino, y_teste = dataset.divisao_treino_teste(X, y)
    assert len(X_treino) == 4000 and len(X_teste) == 1000
    assert set(X_treino.index).isdisjoint(set(X_teste.index))
    assert len(X_treino) + len(X_teste) == len(X)
    assert abs(float(y_treino.mean()) - float(y_teste.mean())) < 0.01


def test_divisao_treino_teste_e_reprodutivel(base_oficial):
    X, y = dataset.separar_features_e_alvo(base_oficial)
    primeira = dataset.divisao_treino_teste(X, y)
    segunda = dataset.divisao_treino_teste(X, y)
    pd.testing.assert_frame_equal(primeira[1], segunda[1])
    pd.testing.assert_series_equal(primeira[3], segunda[3])


def test_divisao_treino_teste_rejeita_tamanho_invalido(base_oficial):
    X, y = dataset.separar_features_e_alvo(base_oficial)
    with pytest.raises(dataset.ErroDeDados):
        dataset.divisao_treino_teste(X, y, tamanho_teste=1.5)


def test_faixas_observadas_cobrem_todas_as_features(base_oficial):
    faixas = dataset.faixas_observadas(base_oficial)
    assert set(faixas) == set(dataset.COLUNAS_PREDITORAS)
    for coluna, (minimo, maximo) in faixas.items():
        assert minimo <= maximo, coluna
    assert faixas["idade_media"][0] == pytest.approx(18.0)
    assert faixas["idade_media"][1] <= 90.0


def test_base_oficial_e_deterministica_entre_importacoes(raiz_do_projeto, base_oficial):
    caminho = Path(raiz_do_projeto) / "data" / "base_dados.py"
    conteudo_original = caminho.read_text(encoding="utf-8")
    try:
        terceira = dataset.carregar_base_oficial(raiz=raiz_do_projeto)
        pd.testing.assert_frame_equal(base_oficial, terceira)
    finally:
        # O material oficial nunca é alterado; a checagem garante que segue intacto.
        assert caminho.read_text(encoding="utf-8") == conteudo_original
