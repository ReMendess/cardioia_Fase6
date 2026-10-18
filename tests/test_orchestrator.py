"""Testes do orquestrador e do pipeline completo (``src/cardioia/agents/orchestrator.py``).

O fluxo é executado de verdade com o SDK: ``Runner.run_sync``, *handoffs* e *function tools*
reais. Quem decide a rota é um ``Model`` local (``ModeloComRoteiro``), porque o ambiente de
teste não usa a API da OpenAI — limitação declarada no relatório e em
``tests/test_integracao_real.py``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("agents", reason="OpenAI Agents SDK nao instalado")

from cardioia.agents import modelo_simulado as simulacao  # noqa: E402
from cardioia.agents import orchestrator  # noqa: E402
from cardioia.agents.contexto import NOME_PLANEJADOR_DE_RESPOSTA  # noqa: E402
from cardioia.ml import predict  # noqa: E402
from cardioia.schemas import risk_result  # noqa: E402

HANDOFFS_ESPERADOS = [
    "Orquestrador -> Analista de Risco",
    "Analista de Risco -> Especialista em Protocolos",
    "Especialista em Protocolos -> Planejador de Resposta",
]

AGENTES_DO_FLUXO = {
    "Orquestrador",
    "Analista de Risco",
    "Especialista em Protocolos",
    "Planejador de Resposta",
}

IDS_ESPERADOS_PACIENTE_AULA = {
    "carga_do_sistema_acima_de_85",
    "disponibilidade_abaixo_de_30",
    "densidade_de_eventos_acima_de_6",
    "risco_sistemico_alto",
}

INDICADORES_SATISFEITOS = {
    "carga_sistema > 80",
    "tempo_espera > 60",
    "spo2_media < 92",
    "fc_media > 110",
    "eventos_24h > 4",
    "disponibilidade_recursos < 40",
}


def _executar(paciente, modelo_temporario, raiz_do_projeto, **extras):
    return orchestrator.executar_pipeline(
        paciente,
        raiz=raiz_do_projeto,
        pasta_modelos=modelo_temporario["pasta"],
        **extras,
    )


def _predicao_esperada(paciente, modelo_temporario, raiz_do_projeto):
    return predict.prever_pico_risco(
        paciente, pasta_modelos=modelo_temporario["pasta"], raiz=raiz_do_projeto
    )


# --------------------------------------------------------------------------- #
# Fluxo completo (feliz)
# --------------------------------------------------------------------------- #


def test_pipeline_conclui_com_o_paciente_da_aula(modelo_temporario, paciente_da_aula, raiz_do_projeto):
    resposta = _executar(paciente_da_aula, modelo_temporario, raiz_do_projeto)
    assert resposta.status is risk_result.StatusDaExecucao.CONCLUIDO
    assert resposta.concluida is True
    assert resposta.erros == []


def test_pipeline_passa_pelos_quatro_agentes(modelo_temporario, paciente_da_aula, raiz_do_projeto):
    resposta = _executar(paciente_da_aula, modelo_temporario, raiz_do_projeto)
    assert {passo.agente for passo in resposta.passos} == AGENTES_DO_FLUXO


def test_handoffs_acontecem_na_ordem_do_fluxo(modelo_temporario, paciente_da_aula, raiz_do_projeto):
    resposta = _executar(paciente_da_aula, modelo_temporario, raiz_do_projeto)
    assert resposta.handoffs_realizados == HANDOFFS_ESPERADOS


def test_agente_final_e_o_planejador(modelo_temporario, paciente_da_aula, raiz_do_projeto):
    resposta = _executar(paciente_da_aula, modelo_temporario, raiz_do_projeto)
    assert resposta.agente_final == NOME_PLANEJADOR_DE_RESPOSTA


def test_predicao_do_contrato_vem_do_modelo_de_ml(modelo_temporario, paciente_da_aula, raiz_do_projeto):
    esperado = _predicao_esperada(paciente_da_aula, modelo_temporario, raiz_do_projeto)
    resposta = _executar(paciente_da_aula, modelo_temporario, raiz_do_projeto)

    assert resposta.diagnostico is not None
    assert resposta.diagnostico.probabilidade_risco == esperado["probabilidade_risco"]
    assert resposta.diagnostico.pico_risco_previsto == esperado["pico_risco_previsto"]
    assert resposta.diagnostico.classificacao_risco == esperado["classificacao_risco"]
    assert resposta.diagnostico.modelo_utilizado == esperado["modelo_utilizado"]
    assert resposta.diagnostico.limiar_decisao == esperado["limiar_decisao"]


def test_probabilidade_e_preservada_sem_arredondamento(modelo_temporario, paciente_da_aula, raiz_do_projeto):
    esperado = _predicao_esperada(paciente_da_aula, modelo_temporario, raiz_do_projeto)
    resposta = _executar(paciente_da_aula, modelo_temporario, raiz_do_projeto)
    assert resposta.diagnostico is not None
    assert resposta.diagnostico.probabilidade_risco == esperado["probabilidade_risco"]
    serializado = resposta.como_dicionario()["diagnostico"]["probabilidade_risco"]
    assert serializado == esperado["probabilidade_risco"]


def test_protocolos_do_contrato_sao_os_oficiais(modelo_temporario, paciente_da_aula, raiz_do_projeto):
    resposta = _executar(paciente_da_aula, modelo_temporario, raiz_do_projeto)
    assert {protocolo.identificador for protocolo in resposta.protocolos} == IDS_ESPERADOS_PACIENTE_AULA
    for protocolo in resposta.protocolos:
        assert protocolo.texto_oficial.startswith("Se ")
        assert protocolo.valor_observado is not None
        assert protocolo.origem_do_valor


def test_justificativas_cobrem_todos_os_protocolos(modelo_temporario, paciente_da_aula, raiz_do_projeto):
    resposta = _executar(paciente_da_aula, modelo_temporario, raiz_do_projeto)
    assert len(resposta.justificativas) == len(resposta.protocolos)
    for justificativa in resposta.justificativas:
        assert "condicao" in justificativa
        assert "valor observado" in justificativa


def test_indicadores_deterministicos_sao_calculados_em_python(
    modelo_temporario, paciente_da_aula, raiz_do_projeto
):
    resposta = _executar(paciente_da_aula, modelo_temporario, raiz_do_projeto)
    satisfeitos = {indicador.nome for indicador in resposta.indicadores if indicador.satisfeito}
    assert satisfeitos == INDICADORES_SATISFEITOS
    for indicador in resposta.indicadores:
        assert indicador.origem == "dados_do_paciente"


def test_pipeline_sem_risco_nao_aciona_protocolo(modelo_temporario, paciente_sem_risco, raiz_do_projeto):
    resposta = _executar(paciente_sem_risco, modelo_temporario, raiz_do_projeto)
    assert resposta.concluida is True
    assert resposta.protocolos == []
    assert resposta.diagnostico is not None
    assert resposta.diagnostico.pico_risco_previsto == 0


def test_resposta_tem_limitações_e_aviso_de_modo_simulado(
    modelo_temporario, paciente_da_aula, raiz_do_projeto
):
    resposta = _executar(paciente_da_aula, modelo_temporario, raiz_do_projeto)
    assert len(resposta.limitacoes) == len(risk_result.LIMITACOES_PADRAO)
    assert any("SIMULADO" in aviso for aviso in resposta.avisos)


def test_contrato_final_e_serializavel(modelo_temporario, paciente_da_aula, raiz_do_projeto):
    resposta = _executar(paciente_da_aula, modelo_temporario, raiz_do_projeto)
    dados = resposta.como_dicionario()
    assert dados["status"] == "CONCLUIDO"
    assert dados["modo"] == "simulado"
    assert dados["versao_do_sdk"] == orchestrator.versao_do_sdk()
    assert dados["duracao_segundos"] is not None


def test_resumo_textual_tem_o_essencial(modelo_temporario, paciente_da_aula, raiz_do_projeto):
    resumo = _executar(paciente_da_aula, modelo_temporario, raiz_do_projeto).resumo_textual()
    assert "PREDICAO" in resumo
    assert "PROTOCOLOS APLICAVEIS (4)" in resumo
    assert "HANDOFFS REALIZADOS" in resumo
    assert "LIMITACOES" in resumo


# --------------------------------------------------------------------------- #
# Histórico de mensagens (o SDK carrega o histórico entre os turnos)
# --------------------------------------------------------------------------- #


def test_historico_do_sdk_carrega_chamadas_e_resultados(
    modelo_temporario, paciente_da_aula, raiz_do_projeto
):
    roteiro = simulacao.ModeloComRoteiro(roteiro=orchestrator.roteiro_padrao())
    resposta = _executar(
        paciente_da_aula, modelo_temporario, raiz_do_projeto, modelo_simulado=roteiro
    )

    assert resposta.concluida is True
    assert roteiro.problemas == []
    assert len(roteiro.historico) == 6  # 5 passos do fluxo + o turno final do Planejador

    # Turno 1: sem histórico anterior (só a mensagem inicial).
    assert "function_call" not in roteiro.historico[0]
    # Turno 2: o SDK já carrega o handoff anterior no histórico.
    assert "function_call" in roteiro.historico[1]
    # Turno 3 (após a tool do modelo): o resultado da tool está no histórico.
    assert "function_call_output" in roteiro.historico[2]
    # Turno final: o histórico acumulado tem chamadas e resultados das duas tools.
    assert roteiro.historico[-1].count("function_call_output") >= 2


def test_ferramentas_sao_oferecidas_na_etapa_certa(
    modelo_temporario, paciente_da_aula, raiz_do_projeto
):
    roteiro = simulacao.ModeloComRoteiro(roteiro=orchestrator.roteiro_padrao())
    _executar(paciente_da_aula, modelo_temporario, raiz_do_projeto, modelo_simulado=roteiro)

    assert "consultar_modelo_de_risco" in roteiro.chamadas[1]
    assert "consultar_protocolos" in roteiro.chamadas[3]
    assert "Planejador de Resposta" in roteiro.chamadas[4]
    # O último turno recebe o esquema de saída estruturada.
    assert "output_schema=sim" in roteiro.chamadas[-1]


# --------------------------------------------------------------------------- #
# Falhas: o pipeline nunca produz sucesso aparente
# --------------------------------------------------------------------------- #


def test_modelo_ausente_gera_erro_antes_de_qualquer_run(paciente_da_aula, raiz_do_projeto, tmp_path):
    resposta = orchestrator.executar_pipeline(
        paciente_da_aula, raiz=raiz_do_projeto, pasta_modelos=tmp_path
    )
    assert resposta.status is risk_result.StatusDaExecucao.ERRO
    assert resposta.diagnostico is None
    assert resposta.handoffs_realizados == []
    assert any("modelo indisponivel" in erro for erro in resposta.erros)
    assert list(tmp_path.iterdir()) == []  # nunca treina silenciosamente


def test_paciente_invalido_gera_erro(modelo_temporario, raiz_do_projeto, paciente_da_aula):
    invalido = dict(paciente_da_aula)
    invalido.pop("spo2_media")
    resposta = _executar(invalido, modelo_temporario, raiz_do_projeto)
    assert resposta.status is risk_result.StatusDaExecucao.ERRO
    assert any("invalida" in erro for erro in resposta.erros)


def test_falha_da_tool_do_modelo_gera_status_de_erro(
    modelo_temporario, paciente_da_aula, raiz_do_projeto
):
    """Roteiro sem a tool do modelo: o Analista falha e o pipeline NÃO conclui."""
    roteiro = simulacao.ModeloComRoteiro(
        roteiro=[
            simulacao.PassoDoRoteiro("handoff", "Analista de Risco"),
            simulacao.PassoDoRoteiro("handoff", "Especialista em Protocolos"),
            simulacao.PassoDoRoteiro("handoff", "Planejador de Resposta"),
            simulacao.PassoDoRoteiro("final", orchestrator.TEXTO_FINAL_SIMULADO),
        ]
    )
    resposta = _executar(
        paciente_da_aula, modelo_temporario, raiz_do_projeto, modelo_simulado=roteiro
    )

    assert resposta.status in (
        risk_result.StatusDaExecucao.ERRO,
        risk_result.StatusDaExecucao.INCOMPLETO,
    )
    assert resposta.concluida is False
    assert resposta.erros
    assert any("sem predicao validada" in erro for erro in resposta.erros)


def test_handoff_incompleto_gera_status_incompleto(
    modelo_temporario, paciente_da_aula, raiz_do_projeto
):
    """Roteiro que termina antes do Planejador: o fluxo não foi concluído."""
    roteiro = simulacao.ModeloComRoteiro(
        roteiro=[
            simulacao.PassoDoRoteiro("handoff", "Analista de Risco"),
            simulacao.PassoDoRoteiro("tool", "consultar_modelo_de_risco"),
            simulacao.PassoDoRoteiro("final", "resposta sem passar pelos especialistas"),
        ]
    )
    resposta = _executar(
        paciente_da_aula, modelo_temporario, raiz_do_projeto, modelo_simulado=roteiro
    )

    assert resposta.status is risk_result.StatusDaExecucao.INCOMPLETO
    assert resposta.agente_final == "Analista de Risco"
    assert any("Planejador de Resposta" in erro for erro in resposta.erros)
    assert resposta.handoffs_realizados == ["Orquestrador -> Analista de Risco"]


def test_roteiro_inconsistente_e_registrado_como_erro(
    modelo_temporario, paciente_da_aula, raiz_do_projeto
):
    """Handoff para um agente que o SDK não ofereceu: erro explícito no resultado."""
    roteiro = simulacao.ModeloComRoteiro(
        roteiro=[
            simulacao.PassoDoRoteiro("handoff", "Agente Inexistente"),
            simulacao.PassoDoRoteiro("final", "texto qualquer"),
        ]
    )
    resposta = _executar(
        paciente_da_aula, modelo_temporario, raiz_do_projeto, modelo_simulado=roteiro
    )
    assert resposta.concluida is False
    assert any("roteiro simulado" in erro for erro in resposta.erros)


def test_limite_de_turnos_gera_erro_explicito(modelo_temporario, paciente_da_aula, raiz_do_projeto):
    resposta = _executar(paciente_da_aula, modelo_temporario, raiz_do_projeto, max_turnos=1)
    assert resposta.status is risk_result.StatusDaExecucao.ERRO
    assert any("MaxTurnsExceeded" in erro for erro in resposta.erros)


def test_modo_real_sem_chave_gera_erro(
    modelo_temporario, paciente_da_aula, raiz_do_projeto, monkeypatch
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    resposta = _executar(
        paciente_da_aula,
        modelo_temporario,
        raiz_do_projeto,
        modo=risk_result.ModoDeExecucao.REAL,
    )
    assert resposta.status is risk_result.StatusDaExecucao.ERRO
    assert resposta.modo is risk_result.ModoDeExecucao.REAL
    assert any("OPENAI_API_KEY" in erro for erro in resposta.erros)


def test_narrativa_divergente_gera_ressalva(modelo_temporario, paciente_da_aula, raiz_do_projeto):
    """Se o texto do Planejador citar outro número, o valor da tool prevalece."""
    roteiro = simulacao.ModeloComRoteiro(
        roteiro=[
            simulacao.PassoDoRoteiro("handoff", "Analista de Risco"),
            simulacao.PassoDoRoteiro("tool", "consultar_modelo_de_risco"),
            simulacao.PassoDoRoteiro("handoff", "Especialista em Protocolos"),
            simulacao.PassoDoRoteiro("tool", "consultar_protocolos"),
            simulacao.PassoDoRoteiro("handoff", "Planejador de Resposta"),
            simulacao.PassoDoRoteiro(
                "final",
                '{"narrativa": "A probabilidade de pico de risco e 0.1234 segundo o modelo.",'
                ' "justificativas": [], "pontos_de_atencao": []}',
            ),
        ]
    )
    esperado = _predicao_esperada(paciente_da_aula, modelo_temporario, raiz_do_projeto)
    resposta = _executar(
        paciente_da_aula, modelo_temporario, raiz_do_projeto, modelo_simulado=roteiro
    )

    assert resposta.status is risk_result.StatusDaExecucao.CONCLUIDO_COM_RESSALVAS
    assert resposta.diagnostico is not None
    assert resposta.diagnostico.probabilidade_risco == esperado["probabilidade_risco"]
    assert any("divergente" in aviso for aviso in resposta.avisos)


# --------------------------------------------------------------------------- #
# Funções de verificação isoladas
# --------------------------------------------------------------------------- #


def test_verificar_narrativa_sem_numeros_nao_avisa():
    assert orchestrator.verificar_narrativa("texto sem numeros", 0.99) == []


def test_verificar_narrativa_com_numero_correto_nao_avisa():
    assert orchestrator.verificar_narrativa("probabilidade 0.9900", 0.99) == []


def test_verificar_narrativa_sem_probabilidade_disponivel_avisa():
    avisos = orchestrator.verificar_narrativa("probabilidade 0.5000", None)
    assert avisos and "nao foi calculada" in avisos[0]


def test_verificar_handoffs_finais():
    assert orchestrator.verificar_handoffs_finais(NOME_PLANEJADOR_DE_RESPOSTA) == []
    assert orchestrator.verificar_handoffs_finais("Analista de Risco")
    assert orchestrator.verificar_handoffs_finais(None)


def test_roteiro_padrao_tem_seis_passos():
    passos = orchestrator.roteiro_padrao()
    assert [passo.tipo for passo in passos] == [
        "handoff",
        "tool",
        "handoff",
        "tool",
        "handoff",
        "final",
    ]
    assert passos[-1].alvo == orchestrator.TEXTO_FINAL_SIMULADO


def test_texto_final_simulado_nao_contem_numeros():
    """A narrativa simulada não pode introduzir valores (nada de número inventado)."""
    import re

    assert not re.search(r"\d+[.,]\d", orchestrator.TEXTO_FINAL_SIMULADO)