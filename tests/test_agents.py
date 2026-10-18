"""Testes dos agentes, das tools e da configuração dos handoffs.

Os testes **não** chamam a API da OpenAI: o SDK é importado de verdade (``importorskip``),
as *function tools* são invocadas de verdade e os handoffs são verificados na estrutura dos
agentes. A execução completa do fluxo (com handoffs realmente ocorrendo) está em
``tests/test_orchestrator.py``.
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("agents", reason="OpenAI Agents SDK nao instalado")

from agents import Agent  # noqa: E402
from agents.handoffs import Handoff  # noqa: E402
from agents.tool_context import ToolContext  # noqa: E402

from cardioia.agents import orchestrator  # noqa: E402
from cardioia.agents.contexto import ContextoDaExecucao  # noqa: E402
from cardioia.ml import predict  # noqa: E402

NOMES_DOS_AGENTES = [
    "Orquestrador",
    "Analista de Risco",
    "Especialista em Protocolos",
    "Planejador de Resposta",
]


def _invocar_tool(tool, contexto: ContextoDaExecucao, argumentos: str = "{}") -> str:
    """Invoca uma ``FunctionTool`` do SDK como o Runner faria (via ``ToolContext``)."""
    wrapper = ToolContext(
        context=contexto,
        tool_name=tool.name,
        tool_call_id="call_de_teste",
        tool_arguments=argumentos,
    )
    return asyncio.run(tool.on_invoke_tool(wrapper, argumentos))


def _achar_tool(agente, nome: str):
    for tool in agente.tools:
        if getattr(tool, "name", None) == nome:
            return tool
    raise AssertionError(f"Tool '{nome}' nao encontrada no agente '{agente.name}'.")


def _contexto(paciente, pasta_modelos=None, raiz=None) -> ContextoDaExecucao:
    return ContextoDaExecucao(
        paciente=dict(paciente), pasta_modelos=pasta_modelos, raiz=raiz
    )


# --------------------------------------------------------------------------- #
# O SDK não é sombreado pelo pacote `src/agents` do repositório
# --------------------------------------------------------------------------- #


def test_sdk_agents_nao_e_sombreado_pelo_repositorio(raiz_do_projeto):
    """``import agents`` deve resolver para o SDK instalado, nunca para ``src/agents``."""
    import agents

    caminho = getattr(agents, "__file__", "") or ""
    assert str(raiz_do_projeto) not in caminho, (
        f"'agents' resolveu para dentro do repositorio: {caminho}"
    )
    assert hasattr(agents, "Agent")
    assert hasattr(agents, "Runner")
    assert hasattr(agents, "function_tool")


# --------------------------------------------------------------------------- #
# Estrutura dos agentes e handoffs
# --------------------------------------------------------------------------- #


def test_pipeline_tem_os_quatro_agentes():
    assert orchestrator.construir_agentes().nomes() == NOMES_DOS_AGENTES


def test_orquestrador_faz_handoff_para_o_analista():
    agentes = orchestrator.construir_agentes()
    destinos = [getattr(h, "name", None) for h in agentes.orquestrador.handoffs]
    assert destinos == ["Analista de Risco"]


def test_analista_faz_handoff_para_o_especialista():
    agentes = orchestrator.construir_agentes()
    destinos = [getattr(h, "name", None) for h in agentes.analista_de_risco.handoffs]
    assert destinos == ["Especialista em Protocolos"]


def test_especialista_faz_handoff_para_o_planejador():
    agentes = orchestrator.construir_agentes()
    destinos = [getattr(h, "name", None) for h in agentes.especialista_em_protocolos.handoffs]
    assert destinos == ["Planejador de Resposta"]


def test_planejador_e_o_agente_final_sem_handoff():
    agentes = orchestrator.construir_agentes()
    assert agentes.planejador_de_resposta.handoffs == []


def test_agentes_de_destino_tem_handoff_description():
    agentes = orchestrator.construir_agentes()
    for agente in (
        agentes.analista_de_risco,
        agentes.especialista_em_protocolos,
        agentes.planejador_de_resposta,
    ):
        assert agente.handoff_description


def test_nome_da_tool_de_handoff_segue_a_convencao_do_sdk():
    """O SDK expõe o handoff como ``transfer_to_<nome_normalizado>``."""
    agente = Agent(name="Analista de Risco", instructions="x")
    assert Handoff.default_tool_name(agente) == "transfer_to_analista_de_risco"


def test_analista_tem_a_tool_do_modelo():
    agentes = orchestrator.construir_agentes()
    assert [tool.name for tool in agentes.analista_de_risco.tools] == ["consultar_modelo_de_risco"]


def test_especialista_tem_a_tool_de_protocolos():
    agentes = orchestrator.construir_agentes()
    nomes = [tool.name for tool in agentes.especialista_em_protocolos.tools]
    assert nomes == ["consultar_protocolos"]


def test_planejador_usa_saida_estruturada():
    agentes = orchestrator.construir_agentes()
    tipo = agentes.planejador_de_resposta.output_type
    assert tipo is not None
    assert tipo.__name__ == "ContribuicaoDoPlanejador"


def test_orquestrador_nao_tem_tools_proprias():
    """O Orquestrador coordena; quem calcula são o Analista e o Especialista."""
    assert orchestrator.construir_agentes().orquestrador.tools == []


# --------------------------------------------------------------------------- #
# Tool do modelo (Agente 1)
# --------------------------------------------------------------------------- #


def test_tool_do_modelo_devolve_a_probabilidade_real(
    modelo_temporario, paciente_da_aula, raiz_do_projeto
):
    agentes = orchestrator.construir_agentes()
    tool = _achar_tool(agentes.analista_de_risco, "consultar_modelo_de_risco")
    contexto = _contexto(
        paciente_da_aula, pasta_modelos=modelo_temporario["pasta"], raiz=raiz_do_projeto
    )

    saida = _invocar_tool(tool, contexto)
    esperado = predict.prever_pico_risco(
        paciente_da_aula, pasta_modelos=modelo_temporario["pasta"], raiz=raiz_do_projeto
    )

    assert contexto.predicao is not None
    assert contexto.predicao["probabilidade_risco"] == esperado["probabilidade_risco"]
    assert contexto.predicao["pico_risco_previsto"] == esperado["pico_risco_previsto"]
    assert f"{esperado['probabilidade_risco']:.4f}" in saida


def test_tool_do_modelo_nao_altera_os_dados_do_paciente(
    modelo_temporario, paciente_da_aula, raiz_do_projeto
):
    agentes = orchestrator.construir_agentes()
    tool = _achar_tool(agentes.analista_de_risco, "consultar_modelo_de_risco")
    contexto = _contexto(
        paciente_da_aula, pasta_modelos=modelo_temporario["pasta"], raiz=raiz_do_projeto
    )
    antes = dict(contexto.paciente)

    _invocar_tool(tool, contexto)

    assert contexto.paciente == antes
    assert contexto.predicao is not None


def test_tool_do_modelo_sem_artefato_registra_erro_sem_derrubar(paciente_da_aula, tmp_path):
    agentes = orchestrator.construir_agentes()
    tool = _achar_tool(agentes.analista_de_risco, "consultar_modelo_de_risco")
    contexto = _contexto(paciente_da_aula, pasta_modelos=tmp_path)

    saida = _invocar_tool(tool, contexto)

    assert "FALHA" in saida
    assert "python -m src.cardioia.ml.train" in saida
    assert contexto.predicao is None
    assert any("modelo ausente" in erro for erro in contexto.erros)
    assert list(tmp_path.iterdir()) == []  # nada foi criado nem treinado


def test_tool_do_modelo_registra_avisos_de_faixa(
    modelo_temporario, paciente_da_aula, raiz_do_projeto
):
    agentes = orchestrator.construir_agentes()
    tool = _achar_tool(agentes.analista_de_risco, "consultar_modelo_de_risco")
    contexto = _contexto(
        paciente_da_aula, pasta_modelos=modelo_temporario["pasta"], raiz=raiz_do_projeto
    )
    _invocar_tool(tool, contexto)
    # O modelo da fixture treina em 400 linhas: a faixa observada é mais estreita.
    assert any("aviso" in passo.detalhe for passo in contexto.passos)


# --------------------------------------------------------------------------- #
# Tool de protocolos (Agente 2)
# --------------------------------------------------------------------------- #


def test_tool_de_protocolos_exige_predicao_no_contexto(
    modelo_temporario, paciente_da_aula, raiz_do_projeto
):
    agentes = orchestrator.construir_agentes()
    tool = _achar_tool(agentes.especialista_em_protocolos, "consultar_protocolos")
    contexto = _contexto(
        paciente_da_aula, pasta_modelos=modelo_temporario["pasta"], raiz=raiz_do_projeto
    )

    saida = _invocar_tool(tool, contexto)

    assert "FALHA" in saida
    assert "predicao" in saida.lower()
    assert contexto.avaliacao_de_protocolos is None
    assert contexto.erros


def test_tool_de_protocolos_usa_a_classe_do_modelo(
    modelo_temporario, paciente_da_aula, raiz_do_projeto
):
    agentes = orchestrator.construir_agentes()
    contexto = _contexto(
        paciente_da_aula, pasta_modelos=modelo_temporario["pasta"], raiz=raiz_do_projeto
    )
    _invocar_tool(_achar_tool(agentes.analista_de_risco, "consultar_modelo_de_risco"), contexto)
    assert contexto.tem_predicao

    saida = _invocar_tool(
        _achar_tool(agentes.especialista_em_protocolos, "consultar_protocolos"), contexto
    )

    assert contexto.avaliacao_de_protocolos is not None
    identificadores = set(contexto.avaliacao_de_protocolos.identificadores)
    assert "carga_do_sistema_acima_de_85" in identificadores
    assert "disponibilidade_abaixo_de_30" in identificadores
    assert "densidade_de_eventos_acima_de_6" in identificadores
    # A regra de risco sistêmico usa a classe prevista (1 para o paciente da aula).
    assert "risco_sistemico_alto" in identificadores
    assert "Se carga do sistema > 85%" in saida
    assert "valor observado: 92" in saida


def test_tool_de_protocolos_sem_risco_nao_sugere_nada(
    modelo_temporario, paciente_sem_risco, raiz_do_projeto
):
    agentes = orchestrator.construir_agentes()
    contexto = _contexto(
        paciente_sem_risco, pasta_modelos=modelo_temporario["pasta"], raiz=raiz_do_projeto
    )
    contexto.predicao = {
        "pico_risco_previsto": 0,
        "probabilidade_risco": 0.01,
        "classificacao_risco": "sem_pico_de_risco",
    }
    saida = _invocar_tool(
        _achar_tool(agentes.especialista_em_protocolos, "consultar_protocolos"), contexto
    )
    assert contexto.avaliacao_de_protocolos is not None
    assert contexto.avaliacao_de_protocolos.quantidade == 0
    assert "0 de 10 protocolos" in saida


def test_tool_de_protocolos_com_dados_invalidos_registra_erro(
    modelo_temporario, paciente_da_aula, raiz_do_projeto
):
    agentes = orchestrator.construir_agentes()
    incompleto = dict(paciente_da_aula)
    incompleto.pop("spo2_media")
    contexto = _contexto(
        incompleto, pasta_modelos=modelo_temporario["pasta"], raiz=raiz_do_projeto
    )
    contexto.predicao = {"pico_risco_previsto": 1, "probabilidade_risco": 0.9}

    saida = _invocar_tool(
        _achar_tool(agentes.especialista_em_protocolos, "consultar_protocolos"), contexto
    )

    assert "FALHA" in saida
    assert contexto.erros
    assert contexto.avaliacao_de_protocolos is None


# --------------------------------------------------------------------------- #
# Planejador (Agente 3): contrato da saída em texto
# --------------------------------------------------------------------------- #


def test_contribuicao_do_planejador_exige_narrativa():
    from pydantic import ValidationError

    from cardioia.agents.response_planner import ContribuicaoDoPlanejador

    with pytest.raises(ValidationError):
        ContribuicaoDoPlanejador()
    contribuicao = ContribuicaoDoPlanejador(narrativa="texto")
    assert contribuicao.justificativas == []
    assert contribuicao.pontos_de_atencao == []


def test_contribuicao_rejeita_campos_desconhecidos():
    from pydantic import ValidationError

    from cardioia.agents.response_planner import ContribuicaoDoPlanejador

    with pytest.raises(ValidationError):
        ContribuicaoDoPlanejador(narrativa="x", probabilidade=0.99)


def test_narrativa_como_texto_aceita_tipos_previstos():
    from cardioia.agents.response_planner import ContribuicaoDoPlanejador, narrativa_como_texto

    contribuicao = ContribuicaoDoPlanejador(
        narrativa="resumo", justificativas=["j1"], pontos_de_atencao=["a1"]
    )
    assert narrativa_como_texto(contribuicao) == ("resumo", ["j1"], ["a1"])
    assert narrativa_como_texto({"narrativa": "d", "justificativas": ["j"]}) == ("d", ["j"], [])
    assert narrativa_como_texto("texto puro") == ("texto puro", [], [])
    assert narrativa_como_texto(None) == (None, [], [])


# --------------------------------------------------------------------------- #
# Contexto compartilhado e trilha de execução
# --------------------------------------------------------------------------- #


def test_contexto_registra_passos_e_erros():
    from cardioia.schemas.risk_result import TipoDePasso

    contexto = _contexto({"carga_sistema": 92.0})
    contexto.registrar(TipoDePasso.HANDOFF, "A", "handoff para B")
    contexto.registrar(TipoDePasso.ERRO, "B", "algo falhou", sucesso=False)

    assert [passo.tipo for passo in contexto.passos] == [TipoDePasso.HANDOFF, TipoDePasso.ERRO]
    assert len(contexto.passos_do_contrato) == 2
    assert contexto.erros == ["[B] algo falhou"]


def test_contexto_guarda_apenas_os_campos_do_contrato():
    """O contexto não acumula dados além do necessário (base sintética, campos do contrato)."""
    from cardioia.ml import dataset

    contexto = _contexto({chave: 1.0 for chave in dataset.COLUNAS_PREDITORAS})
    assert set(contexto.paciente) == set(dataset.COLUNAS_PREDITORAS)


def test_trilha_de_execucao_registra_handoff_no_contexto():
    from cardioia.agents.contexto import TrilhaDeExecucao
    from cardioia.schemas.risk_result import TipoDePasso

    contexto = _contexto({"carga_sistema": 92.0})
    wrapper = type("WrapperFalso", (), {"context": contexto})()
    trilha = TrilhaDeExecucao()

    class AgenteFalso:
        def __init__(self, nome: str) -> None:
            self.name = nome

    asyncio.run(trilha.on_handoff(wrapper, AgenteFalso("A"), AgenteFalso("B")))

    assert contexto.handoffs == [("A", "B")]
    assert contexto.handoffs_realizados == ["A -> B"]
    assert contexto.passos[0].tipo is TipoDePasso.HANDOFF