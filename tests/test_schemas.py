"""Testes dos contratos Pydantic (``src/cardioia/schemas/risk_result.py``).

Cobrem: probabilidades válidas e inválidas, classes permitidas, campos obrigatórios,
estrutura dos protocolos, coerência entre status e conteúdo, proibição de campos
desconhecidos e serialização JSON-safe.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from cardioia.ml import predict
from cardioia.schemas import risk_result as rr


def _diagnostico(**alteracoes) -> dict:
    base = {
        "probabilidade_risco": 0.99,
        "pico_risco_previsto": 1,
        "classificacao_risco": "pico_de_risco",
        "modelo_utilizado": "Random Forest",
        "classe_do_estimador": "RandomForestClassifier",
        "limiar_decisao": 0.5,
        "regra_de_decisao": "probabilidade_risco >= limiar (0.5)",
        "features_utilizadas": list(predict.COLUNAS_PREDITORAS),
    }
    base.update(alteracoes)
    return base


def _protocolo(**alteracoes) -> dict:
    base = {
        "ordem": 2,
        "identificador": "carga_do_sistema_acima_de_85",
        "texto_oficial": "Se carga do sistema > 85% → abrir unidades temporárias",
        "acao": "abrir unidades temporárias",
        "condicao_satisfeita": "carga_sistema > 85",
        "operador": ">",
        "limiar": 85.0,
        "valor_observado": 92.0,
        "origem_do_valor": "dados_do_paciente",
    }
    base.update(alteracoes)
    return base


# --------------------------------------------------------------------------- #
# Probabilidade
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("probabilidade", [0.0, 0.5, 0.99, 1.0])
def test_probabilidade_valida_e_aceita(probabilidade):
    diagnostico = rr.DiagnosticoDaPredicao(**_diagnostico(probabilidade_risco=probabilidade))
    assert diagnostico.probabilidade_risco == probabilidade


@pytest.mark.parametrize("probabilidade", [-0.01, 1.01, 2.0, -1.0])
def test_probabilidade_fora_do_intervalo_e_rejeitada(probabilidade):
    with pytest.raises(ValidationError):
        rr.DiagnosticoDaPredicao(**_diagnostico(probabilidade_risco=probabilidade))


@pytest.mark.parametrize("probabilidade", [float("nan"), float("inf"), -float("inf")])
def test_probabilidade_nao_finita_e_rejeitada(probabilidade):
    with pytest.raises(ValidationError):
        rr.DiagnosticoDaPredicao(**_diagnostico(probabilidade_risco=probabilidade))


def test_probabilidade_ausente_e_permitida_como_none():
    diagnostico = rr.DiagnosticoDaPredicao(**_diagnostico(probabilidade_risco=None))
    assert diagnostico.probabilidade_risco is None


def test_probabilidade_nao_e_arredondada_pelo_contrato():
    valor = 0.987654321
    diagnostico = rr.DiagnosticoDaPredicao(**_diagnostico(probabilidade_risco=valor))
    assert diagnostico.probabilidade_risco == valor


# --------------------------------------------------------------------------- #
# Classe prevista e classificação
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("classe", "classificacao"),
    [(0, "sem_pico_de_risco"), (1, "pico_de_risco")],
)
def test_classes_permitidas(classe, classificacao):
    diagnostico = rr.DiagnosticoDaPredicao(
        **_diagnostico(pico_risco_previsto=classe, classificacao_risco=classificacao)
    )
    assert diagnostico.pico_risco_previsto == classe


@pytest.mark.parametrize("classe", [2, -1, "1", True, None])
def test_classe_invalida_e_rejeitada(classe):
    with pytest.raises(ValidationError):
        rr.DiagnosticoDaPredicao(**_diagnostico(pico_risco_previsto=classe))


def test_classificacao_incoerente_com_a_classe_e_rejeitada():
    with pytest.raises(ValidationError):
        rr.DiagnosticoDaPredicao(
            **_diagnostico(pico_risco_previsto=0, classificacao_risco="pico_de_risco")
        )


def test_classificacao_textual_desconhecida_e_rejeitada():
    with pytest.raises(ValidationError):
        rr.DiagnosticoDaPredicao(**_diagnostico(classificacao_risco="risco_alto"))


def test_rotulos_do_contrato_batem_com_o_modulo_de_ml():
    assert rr.ROTULOS == dict(predict.ROTULOS)
    assert rr.CLASSE_POSITIVA == predict.CLASSE_POSITIVA


# --------------------------------------------------------------------------- #
# Campos obrigatórios
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "campo",
    [
        "pico_risco_previsto",
        "classificacao_risco",
        "modelo_utilizado",
        "classe_do_estimador",
        "limiar_decisao",
        "regra_de_decisao",
        "features_utilizadas",
    ],
)
def test_campo_obrigatorio_do_diagnostico(campo):
    dados = _diagnostico()
    dados.pop(campo)
    with pytest.raises(ValidationError):
        rr.DiagnosticoDaPredicao(**dados)


def test_campo_desconhecido_e_rejeitado():
    with pytest.raises(ValidationError):
        rr.DiagnosticoDaPredicao(**_diagnostico(campo_inventado=1))


def test_limiar_fora_do_intervalo_e_rejeitado():
    with pytest.raises(ValidationError):
        rr.DiagnosticoDaPredicao(**_diagnostico(limiar_decisao=1.5))


def test_features_vazias_sao_rejeitadas():
    with pytest.raises(ValidationError):
        rr.DiagnosticoDaPredicao(**_diagnostico(features_utilizadas=[]))


# --------------------------------------------------------------------------- #
# Estrutura dos protocolos
# --------------------------------------------------------------------------- #


def test_protocolo_valido():
    protocolo = rr.ProtocoloAplicavel(**_protocolo())
    assert protocolo.ordem == 2
    assert protocolo.usa_modelo is False


@pytest.mark.parametrize("ordem", [0, 11, -1])
def test_ordem_do_protocolo_fora_da_base_e_rejeitada(ordem):
    with pytest.raises(ValidationError):
        rr.ProtocoloAplicavel(**_protocolo(ordem=ordem))


def test_ordens_1_e_10_sao_aceitas():
    assert rr.ProtocoloAplicavel(**_protocolo(ordem=1)).ordem == 1
    assert rr.ProtocoloAplicavel(**_protocolo(ordem=10)).ordem == 10


def test_operador_do_protocolo_e_restrito():
    with pytest.raises(ValidationError):
        rr.ProtocoloAplicavel(**_protocolo(operador=">="))


def test_operador_igualdade_e_aceito_para_risco_sistemico():
    protocolo = rr.ProtocoloAplicavel(
        **_protocolo(
            ordem=10,
            identificador="risco_sistemico_alto",
            operador="==",
            limiar=1.0,
            valor_observado=1.0,
            usa_modelo=True,
        )
    )
    assert protocolo.usa_modelo is True


@pytest.mark.parametrize(
    "campo",
    ["identificador", "texto_oficial", "acao", "condicao_satisfeita", "origem_do_valor"],
)
def test_campos_obrigatorios_do_protocolo(campo):
    dados = _protocolo()
    dados.pop(campo)
    with pytest.raises(ValidationError):
        rr.ProtocoloAplicavel(**dados)


def test_texto_vazio_no_protocolo_e_rejeitado():
    with pytest.raises(ValidationError):
        rr.ProtocoloAplicavel(**_protocolo(texto_oficial=""))


# --------------------------------------------------------------------------- #
# Indicadores determinísticos
# --------------------------------------------------------------------------- #


def test_indicador_valido():
    indicador = rr.IndicadorDeterministico(
        nome="carga_sistema > 80",
        coluna="carga_sistema",
        operador=">",
        limiar=80.0,
        valor_observado=92.0,
        satisfeito=True,
        origem="dados_do_paciente",
    )
    assert indicador.satisfeito is True
    assert indicador.coluna == "carga_sistema"


def test_indicador_com_operador_invalido_e_rejeitado():
    with pytest.raises(ValidationError):
        rr.IndicadorDeterministico(
            nome="x",
            coluna="carga_sistema",
            operador="!=",
            limiar=1.0,
            valor_observado=2.0,
            satisfeito=True,
            origem="dados_do_paciente",
        )


# --------------------------------------------------------------------------- #
# Resposta final: coerência entre status e conteúdo
# --------------------------------------------------------------------------- #


def test_resposta_concluida_exige_diagnostico():
    with pytest.raises(ValidationError):
        rr.RespostaDaExecucao(id_execucao="a" * 8, status=rr.StatusDaExecucao.CONCLUIDO)


def test_resposta_concluida_com_diagnostico_e_aceita():
    resposta = rr.RespostaDaExecucao(
        id_execucao="a" * 8,
        status=rr.StatusDaExecucao.CONCLUIDO,
        diagnostico=rr.DiagnosticoDaPredicao(**_diagnostico()),
        protocolos=[rr.ProtocoloAplicavel(**_protocolo())],
        justificativas=["justificativa objetiva"],
    )
    assert resposta.concluida is True
    assert len(resposta.protocolos) == 1


def test_resposta_concluida_nao_pode_conter_erros():
    with pytest.raises(ValidationError):
        rr.RespostaDaExecucao(
            id_execucao="a" * 8,
            status=rr.StatusDaExecucao.CONCLUIDO,
            diagnostico=rr.DiagnosticoDaPredicao(**_diagnostico()),
            erros=["algo falhou"],
        )


@pytest.mark.parametrize("status", [rr.StatusDaExecucao.ERRO, rr.StatusDaExecucao.INCOMPLETO])
def test_status_de_falha_exige_erro_descrito(status):
    with pytest.raises(ValidationError):
        rr.RespostaDaExecucao(id_execucao="a" * 8, status=status)
    resposta = rr.RespostaDaExecucao(
        id_execucao="a" * 8, status=status, erros=["motivo da falha"]
    )
    assert resposta.concluida is False


def test_ressalva_permite_erro_com_predicao():
    resposta = rr.RespostaDaExecucao(
        id_execucao="a" * 8,
        status=rr.StatusDaExecucao.CONCLUIDO_COM_RESSALVAS,
        diagnostico=rr.DiagnosticoDaPredicao(**_diagnostico()),
        erros=["um aviso virou erro"],
    )
    assert resposta.status is rr.StatusDaExecucao.CONCLUIDO_COM_RESSALVAS


def test_id_de_execucao_curto_e_rejeitado():
    with pytest.raises(ValidationError):
        rr.RespostaDaExecucao(id_execucao="abc", status=rr.StatusDaExecucao.ERRO, erros=["x"])


def test_campo_desconhecido_na_resposta_e_rejeitado():
    with pytest.raises(ValidationError):
        rr.RespostaDaExecucao(
            id_execucao="a" * 8,
            status=rr.StatusDaExecucao.ERRO,
            erros=["x"],
            campo_inventado=True,
        )


def test_limitacoes_padrao_estao_presentes():
    resposta = rr.RespostaDaExecucao(
        id_execucao="a" * 8, status=rr.StatusDaExecucao.ERRO, erros=["x"]
    )
    assert len(resposta.limitacoes) == len(rr.LIMITACOES_PADRAO)
    assert any("SINTETICOS" in item for item in resposta.limitacoes)
    assert any("diagnostico" in item for item in resposta.limitacoes)


def test_contrato_de_exemplo_e_valido():
    resposta = rr.RespostaDaExecucao(**rr.contrato_de_exemplo())
    assert resposta.status is rr.StatusDaExecucao.INCOMPLETO


# --------------------------------------------------------------------------- #
# Serialização e apresentação
# --------------------------------------------------------------------------- #


def test_resposta_e_serializavel_em_json():
    resposta = rr.RespostaDaExecucao(
        id_execucao="a" * 8,
        status=rr.StatusDaExecucao.CONCLUIDO,
        diagnostico=rr.DiagnosticoDaPredicao(**_diagnostico()),
        protocolos=[rr.ProtocoloAplicavel(**_protocolo())],
        justificativas=["justificativa"],
        narrativa="texto",
    )
    recarregado = json.loads(json.dumps(resposta.como_dicionario(), ensure_ascii=False))
    assert recarregado["status"] == "CONCLUIDO"
    assert recarregado["diagnostico"]["probabilidade_risco"] == 0.99
    assert recarregado["protocolos"][0]["texto_oficial"].startswith("Se carga do sistema")


def test_resumo_textual_do_modo_simulado_omite_ressalvas():
    """No modo simulado o texto sai enxuto: sem LIMITACOES, AVISOS e ERROS (decisão do autor)."""
    resposta = rr.RespostaDaExecucao(
        id_execucao="a" * 8,
        status=rr.StatusDaExecucao.CONCLUIDO,
        diagnostico=rr.DiagnosticoDaPredicao(**_diagnostico()),
        protocolos=[rr.ProtocoloAplicavel(**_protocolo())],
        justificativas=["justificativa objetiva"],
        avisos=["aviso qualquer do modo simulado"],
    )
    resumo = resposta.resumo_textual()
    assert "0.9900" in resumo  # arredondamento apenas na apresentação
    assert "PROTOCOLOS APLICAVEIS (1)" in resumo
    assert "JUSTIFICATIVAS" in resumo
    for secao in rr.SECOES_OCULTAS_NO_MODO_SIMULADO:
        assert secao not in resumo, f"secao impressa no modo simulado: {secao}"
    # A supressão é só de apresentação: o conteúdo segue no contrato (auditoria/JSON).
    assert list(resposta.limitacoes) == list(rr.LIMITACOES_PADRAO)
    assert resposta.avisos == ["aviso qualquer do modo simulado"]


def test_resumo_textual_do_modo_real_mostra_ressalvas():
    """No modo real as três seções continuam sendo impressas."""
    resposta = rr.RespostaDaExecucao(
        id_execucao="a" * 8,
        status=rr.StatusDaExecucao.CONCLUIDO_COM_RESSALVAS,
        modo=rr.ModoDeExecucao.REAL,
        diagnostico=rr.DiagnosticoDaPredicao(**_diagnostico()),
        avisos=["narrativa citou numero divergente"],
        erros=["falha registrada com predicao valida"],
    )
    assert resposta.mostra_ressalvas is True
    resumo = resposta.resumo_textual()
    assert "AVISOS" in resumo
    assert "ERROS" in resumo
    assert "LIMITACOES" in resumo


def test_resumo_textual_de_falha_no_modo_simulado_mostra_os_erros():
    """Exceção da regra: execução que não concluiu nunca esconde o motivo da falha."""
    resposta = rr.RespostaDaExecucao(
        id_execucao="a" * 8, status=rr.StatusDaExecucao.ERRO, erros=["motivo real da falha"]
    )
    assert resposta.modo is rr.ModoDeExecucao.SIMULADO
    assert resposta.mostra_ressalvas is True
    resumo = resposta.resumo_textual()
    assert "ERROS" in resumo
    assert "motivo real da falha" in resumo
    assert "LIMITACOES" in resumo


def test_resumo_textual_sem_predicao_omite_a_secao():
    resposta = rr.RespostaDaExecucao(
        id_execucao="a" * 8, status=rr.StatusDaExecucao.ERRO, erros=["falha"]
    )
    resumo = resposta.resumo_textual()
    assert "PREDICAO" not in resumo
    assert "ERROS" in resumo