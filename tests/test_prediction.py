"""Testes da camada de inferência (``src/cardioia/ml/predict.py``).

Cobrem: carregamento do modelo salvo, previsão de um caso válido, faixa da probabilidade,
consistência entre o contrato de entrada e as features do modelo, campos ausentes,
valores inválidos, erro explícito quando o artefato não existe e estabilidade da previsão
após salvar/recarregar.

O modelo usado vem da fixture ``modelo_temporario`` (treinado em ``tmp_path``), então a
suíte **não** depende do ``.pkl`` versionado. Nenhum teste acessa a internet.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from cardioia.ml import dataset, predict, train


def _caso(paciente_da_aula, **alteracoes) -> dict:
    """Cópia do caso da aula, opcionalmente com campos alterados/removidos."""
    caso = dict(paciente_da_aula)
    for chave, valor in alteracoes.items():
        if valor is None:
            caso.pop(chave, None)
        else:
            caso[chave] = valor
    return caso


# --------------------------------------------------------------------------- #
# Carregamento dos artefatos
# --------------------------------------------------------------------------- #


def test_carregar_modelo_e_metadados(modelo_temporario):
    modelo = predict.carregar_modelo(modelo_temporario["caminho_modelo"])
    metadados = predict.carregar_metadados(modelo_temporario["caminho_metadados"])
    assert hasattr(modelo, "predict")
    assert metadados["variavel_alvo"] == dataset.VARIAVEL_ALVO
    assert metadados["features"] == list(dataset.COLUNAS_PREDITORAS)
    assert metadados["limiar_de_decisao"] == predict.LIMIAR_DE_DECISAO


def test_metadados_do_modelo_contem_rastreabilidade(modelo_temporario):
    metadados = modelo_temporario["metadados"]
    assert metadados["sha256_da_base_usada"]
    assert metadados["semente"] == dataset.SEMENTE
    assert metadados["split_estratificado"] is True
    assert metadados["amostras_treino"] + metadados["amostras_teste"] == metadados["amostras_total"]
    assert "validacao clinica" in metadados["aviso"]


def test_modelo_ausente_gera_erro_com_instrucao(tmp_path):
    with pytest.raises(predict.ModeloNaoEncontradoError) as erro:
        predict.carregar_modelo(pasta_modelos=tmp_path)
    mensagem = str(erro.value)
    assert "modelo_pico_risco.pkl" in mensagem
    assert predict.COMANDO_DE_TREINO in mensagem


def test_metadados_ausentes_geram_erro(tmp_path):
    with pytest.raises(predict.ModeloNaoEncontradoError):
        predict.carregar_metadados(pasta_modelos=tmp_path)


def test_previsao_sem_artefato_nao_treina_silenciosamente(tmp_path, paciente_da_aula):
    with pytest.raises(predict.ModeloNaoEncontradoError) as erro:
        predict.prever_pico_risco(paciente_da_aula, pasta_modelos=tmp_path)
    assert "python -m src.cardioia.ml.train" in str(erro.value)
    assert list(tmp_path.iterdir()) == []  # nada foi criado


# --------------------------------------------------------------------------- #
# Previsão de um caso válido
# --------------------------------------------------------------------------- #


def test_previsao_do_paciente_da_aula_prevê_pico_de_risco(modelo_temporario, paciente_da_aula):
    resultado = predict.prever_pico_risco(paciente_da_aula, pasta_modelos=modelo_temporario["pasta"])
    assert resultado["pico_risco_previsto"] == 1
    assert resultado["classificacao_risco"] == "pico_de_risco"


def test_probabilidade_fica_entre_0_e_1(modelo_temporario, paciente_da_aula):
    resultado = predict.prever_pico_risco(paciente_da_aula, pasta_modelos=modelo_temporario["pasta"])
    probabilidade = resultado["probabilidade_risco"]
    assert isinstance(probabilidade, float)
    assert 0.0 <= probabilidade <= 1.0


def test_probabilidade_nao_e_arredondada(modelo_temporario, paciente_da_aula):
    entrada = pd.DataFrame([paciente_da_aula])[list(dataset.COLUNAS_PREDITORAS)]
    esperado = float(modelo_temporario["modelo"].predict_proba(entrada)[0][1])
    resultado = predict.prever_pico_risco(paciente_da_aula, pasta_modelos=modelo_temporario["pasta"])
    assert resultado["probabilidade_risco"] == pytest.approx(esperado, abs=0.0)


def test_classificacao_usa_o_limiar_declarado(modelo_temporario, paciente_da_aula):
    pasta = modelo_temporario["pasta"]
    permissivo = predict.prever_pico_risco(paciente_da_aula, pasta_modelos=pasta, limiar=0.0)
    exigente = predict.prever_pico_risco(paciente_da_aula, pasta_modelos=pasta, limiar=0.999999)
    assert permissivo["pico_risco_previsto"] == 1
    assert exigente["pico_risco_previsto"] == 0
    assert permissivo["probabilidade_risco"] == exigente["probabilidade_risco"]
    assert exigente["classificacao_risco"] == "sem_pico_de_risco"


def test_resultado_tem_os_campos_do_contrato(modelo_temporario, paciente_da_aula):
    resultado = predict.prever_pico_risco(paciente_da_aula, pasta_modelos=modelo_temporario["pasta"])
    obrigatorios = {
        "pico_risco_previsto",
        "probabilidade_risco",
        "classificacao_risco",
        "modelo_utilizado",
        "limiar_decisao",
        "regra_de_decisao",
        "features_utilizadas",
        "entrada",
        "origem_do_modelo",
        "avisos",
        "aviso_de_uso",
    }
    assert obrigatorios.issubset(resultado.keys())
    assert _resultado_serializavel(resultado)


def _resultado_serializavel(resultado: dict) -> bool:
    """O resultado precisa ser serializável em JSON (exigência para virar tool de agente)."""
    json.dumps(resultado, ensure_ascii=False)
    return True


# --------------------------------------------------------------------------- #
# Contrato de entrada
# --------------------------------------------------------------------------- #


def test_contrato_de_features_bate_com_o_metadata(modelo_temporario, paciente_da_aula):
    resultado = predict.prever_pico_risco(paciente_da_aula, pasta_modelos=modelo_temporario["pasta"])
    assert resultado["features_utilizadas"] == modelo_temporario["metadados"]["features"]
    assert list(resultado["entrada"]) == resultado["features_utilizadas"]


def test_ordem_das_colunas_nao_importa(modelo_temporario, paciente_da_aula):
    pasta = modelo_temporario["pasta"]
    invertido = {chave: paciente_da_aula[chave] for chave in reversed(list(paciente_da_aula))}
    direto = predict.prever_pico_risco(paciente_da_aula, pasta_modelos=pasta)
    trocado = predict.prever_pico_risco(invertido, pasta_modelos=pasta)
    assert direto["pico_risco_previsto"] == trocado["pico_risco_previsto"]
    assert direto["probabilidade_risco"] == pytest.approx(trocado["probabilidade_risco"])


def test_dataframe_de_uma_linha_e_aceito(modelo_temporario, paciente_da_aula):
    tabela = pd.DataFrame([paciente_da_aula])
    resultado = predict.prever_pico_risco(tabela, pasta_modelos=modelo_temporario["pasta"])
    assert resultado["pico_risco_previsto"] == 1


def test_dataframe_com_varias_linhas_e_rejeitado(modelo_temporario, paciente_da_aula):
    tabela = pd.DataFrame([paciente_da_aula, paciente_da_aula])
    with pytest.raises(predict.PacienteInvalidoError) as erro:
        predict.prever_pico_risco(tabela, pasta_modelos=modelo_temporario["pasta"])
    assert "prever_varios" in str(erro.value)


def test_entrada_de_tipo_errado_e_rejeitada(modelo_temporario):
    with pytest.raises(predict.PacienteInvalidoError):
        predict.prever_pico_risco([62, 92, 75], pasta_modelos=modelo_temporario["pasta"])


def test_feature_ausente_gera_erro_explicito(modelo_temporario, paciente_da_aula):
    caso = _caso(paciente_da_aula, fc_media=None)
    with pytest.raises(predict.PacienteInvalidoError) as erro:
        predict.prever_pico_risco(caso, pasta_modelos=modelo_temporario["pasta"])
    assert "fc_media" in str(erro.value)


def test_todas_as_features_ausentes_geram_erro(modelo_temporario):
    with pytest.raises(predict.PacienteInvalidoError):
        predict.prever_pico_risco({}, pasta_modelos=modelo_temporario["pasta"])


def test_campo_desconhecido_gera_erro(modelo_temporario, paciente_da_aula):
    caso = _caso(paciente_da_aula, campo_inventado=1)
    with pytest.raises(predict.PacienteInvalidoError) as erro:
        predict.prever_pico_risco(caso, pasta_modelos=modelo_temporario["pasta"])
    assert "campo_inventado" in str(erro.value)


def test_coluna_com_espaco_do_material_gera_erro_com_dica(modelo_temporario, paciente_da_aula):
    caso = dict(paciente_da_aula)
    caso["disponibilidade recursos"] = caso.pop("disponibilidade_recursos")
    with pytest.raises(predict.PacienteInvalidoError) as erro:
        predict.prever_pico_risco(caso, pasta_modelos=modelo_temporario["pasta"])
    mensagem = str(erro.value)
    assert "disponibilidade recursos" in mensagem
    assert "disponibilidade_recursos" in mensagem


def test_campo_extra_pode_ser_ignorado_quando_permitido(modelo_temporario, paciente_da_aula):
    caso = _caso(paciente_da_aula, observacao="paciente simulado")
    valores, avisos = predict.validar_paciente(caso, permitir_campos_extras=True)
    assert set(valores) == set(dataset.COLUNAS_PREDITORAS)
    assert avisos == []


def test_permitir_campos_extras_ainda_exige_o_contrato(modelo_temporario, paciente_da_aula):
    caso = _caso(paciente_da_aula, fc_media=None)
    with pytest.raises(predict.PacienteInvalidoError):
        predict.validar_paciente(caso, permitir_campos_extras=True)


# --------------------------------------------------------------------------- #
# Valores inválidos
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "valor",
    ["118", True, float("nan"), float("inf"), -1.0, None],
    ids=["texto", "booleano", "NaN", "infinito", "negativo", "nulo"],
)
def test_valores_invalidos_sao_rejeitados(modelo_temporario, paciente_da_aula, valor):
    caso = _caso(paciente_da_aula, fc_media=valor)
    with pytest.raises(predict.PacienteInvalidoError):
        predict.prever_pico_risco(caso, pasta_modelos=modelo_temporario["pasta"])


def test_valor_negativo_tem_mensagem_especifica(modelo_temporario, paciente_da_aula):
    caso = _caso(paciente_da_aula, eventos_24h=-3)
    with pytest.raises(predict.PacienteInvalidoError) as erro:
        predict.prever_pico_risco(caso, pasta_modelos=modelo_temporario["pasta"])
    assert "negativo" in str(erro.value)


def test_texto_nao_e_convertido_silenciosamente(modelo_temporario, paciente_da_aula):
    caso = _caso(paciente_da_aula, idade_media="62")
    with pytest.raises(predict.PacienteInvalidoError) as erro:
        predict.prever_pico_risco(caso, pasta_modelos=modelo_temporario["pasta"])
    assert "texto" in str(erro.value)


def test_valor_fora_da_faixa_gera_aviso_e_nao_erro(modelo_temporario, paciente_da_aula):
    caso = _caso(paciente_da_aula, fc_media=220.0)
    resultado = predict.prever_pico_risco(caso, pasta_modelos=modelo_temporario["pasta"])
    assert resultado["pico_risco_previsto"] in (0, 1)
    assert any("fora da faixa" in aviso and "fc_media" in aviso for aviso in resultado["avisos"])


def test_caso_dentro_da_faixa_de_projeto_nao_gera_aviso(paciente_da_aula):
    """Sem faixas do metadata, valem as faixas de projeto do gerador: caso da aula é válido."""
    _, avisos = predict.validar_paciente(paciente_da_aula)
    assert avisos == []


def test_faixa_do_treino_pode_ser_mais_estreita_que_a_de_projeto(modelo_temporario, paciente_da_aula):
    """O modelo da fixture foi treinado em 400 linhas: a faixa observada é mais estreita.

    Comportamento documentado: avisar (não recusar) quando o caso sai da faixa vista no
    treino. A decisão não muda por causa do aviso.
    """
    resultado = predict.prever_pico_risco(paciente_da_aula, pasta_modelos=modelo_temporario["pasta"])
    assert resultado["pico_risco_previsto"] == 1
    assert all("fora da faixa observada no treino" in aviso for aviso in resultado["avisos"])


def test_aviso_de_faixa_nao_altera_a_previsao(modelo_temporario, paciente_da_aula):
    pasta = modelo_temporario["pasta"]
    metadados = predict.carregar_metadados(pasta_modelos=pasta)
    modelo = predict.carregar_modelo(pasta_modelos=pasta)

    com_faixas = predict.prever_pico_risco(
        paciente_da_aula, pasta_modelos=pasta, modelo=modelo, metadados=metadados
    )
    sem_faixas = predict.prever_pico_risco(
        paciente_da_aula,
        pasta_modelos=pasta,
        modelo=modelo,
        metadados={chave: valor for chave, valor in metadados.items() if chave != "faixas_esperadas"},
    )
    assert com_faixas["pico_risco_previsto"] == sem_faixas["pico_risco_previsto"] == 1
    assert com_faixas["probabilidade_risco"] == pytest.approx(sem_faixas["probabilidade_risco"])


def test_limiar_invalido_gera_erro(modelo_temporario, paciente_da_aula):
    with pytest.raises(predict.ErroDePrevisao):
        predict.prever_pico_risco(
            paciente_da_aula, pasta_modelos=modelo_temporario["pasta"], limiar=1.5
        )


# --------------------------------------------------------------------------- #
# Persistência e reuso
# --------------------------------------------------------------------------- #


def test_previsao_e_estavel_apos_salvar_e_recarregar(modelo_temporario, paciente_da_aula):
    pasta_nova = modelo_temporario["pasta"] / "recarga"
    artefatos = train.salvar_artefatos(modelo_temporario["resultado"], pasta_modelos=pasta_nova)

    antes = predict.prever_pico_risco(paciente_da_aula, pasta_modelos=modelo_temporario["pasta"])
    depois = predict.prever_pico_risco(paciente_da_aula, pasta_modelos=pasta_nova)
    assert artefatos["caminho_modelo"].is_file()
    assert artefatos["caminho_metadados"].is_file()
    assert antes["pico_risco_previsto"] == depois["pico_risco_previsto"]
    assert antes["probabilidade_risco"] == pytest.approx(depois["probabilidade_risco"])


def test_modelo_do_disco_e_do_memoria_produzem_a_mesma_previsao(modelo_temporario, paciente_da_aula):
    metadados = predict.carregar_metadados(pasta_modelos=modelo_temporario["pasta"])
    modelo_do_disco = predict.carregar_modelo(pasta_modelos=modelo_temporario["pasta"])
    entrada = pd.DataFrame([paciente_da_aula])[list(dataset.COLUNAS_PREDITORAS)]

    do_disco = predict.prever_pico_risco(
        paciente_da_aula,
        pasta_modelos=modelo_temporario["pasta"],
        modelo=modelo_do_disco,
        metadados=metadados,
    )
    em_memoria = predict.prever_pico_risco(
        paciente_da_aula,
        pasta_modelos=modelo_temporario["pasta"],
        modelo=modelo_temporario["modelo"],
        metadados=metadados,
    )
    assert do_disco["pico_risco_previsto"] == em_memoria["pico_risco_previsto"]
    assert do_disco["probabilidade_risco"] == pytest.approx(em_memoria["probabilidade_risco"])
    assert int(modelo_temporario["modelo"].predict(entrada)[0]) == do_disco["pico_risco_previsto"]


def test_prever_varios_processa_uma_lista(modelo_temporario, paciente_da_aula):
    caso_leve = _caso(
        paciente_da_aula,
        carga_sistema=40.0,
        tempo_espera=20.0,
        fc_media=75.0,
        spo2_media=98.0,
        disponibilidade_recursos=85.0,
        eventos_24h=1.0,
    )
    resultados = predict.prever_varios(
        [paciente_da_aula, caso_leve], pasta_modelos=modelo_temporario["pasta"]
    )
    assert len(resultados) == 2
    assert resultados[0]["pico_risco_previsto"] == 1
    assert resultados[1]["pico_risco_previsto"] == 0


def test_resumo_para_agente_descreve_o_resultado(modelo_temporario, paciente_da_aula):
    resultado = predict.prever_pico_risco(paciente_da_aula, pasta_modelos=modelo_temporario["pasta"])
    resumo = predict.resumo_para_agente(resultado)
    assert "pico_risco_previsto=1" in resumo
    assert "probabilidade_risco=" in resumo
    assert "classificacao_risco=pico_de_risco" in resumo
    assert "sintetica_sem_validacao_clinica" in resumo


def test_aviso_de_uso_esta_sempre_presente(modelo_temporario, paciente_da_aula):
    resultado = predict.prever_pico_risco(paciente_da_aula, pasta_modelos=modelo_temporario["pasta"])
    assert "SINTETICOS" in resultado["aviso_de_uso"]
    assert "validacao clinica" in resultado["aviso_de_uso"]


def test_origem_do_modelo_aponta_os_artefatos(modelo_temporario, paciente_da_aula):
    resultado = predict.prever_pico_risco(paciente_da_aula, pasta_modelos=modelo_temporario["pasta"])
    origem = resultado["origem_do_modelo"]
    assert str(modelo_temporario["caminho_modelo"]) == origem["caminho_modelo"]
    assert origem["semente"] == dataset.SEMENTE
    assert origem["sha256_da_base_usada"] == modelo_temporario["metadados"]["sha256_da_base_usada"]
