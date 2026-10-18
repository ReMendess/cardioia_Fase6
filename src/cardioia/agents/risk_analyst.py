"""Agente 1 — Analista de Risco.

Responsabilidade única: **consultar o modelo de ML** e devolver a probabilidade real. O
LLM **não** estima, não arredonda e não inventa probabilidade: ele chama a tool Python
``consultar_modelo_de_risco``, que roda ``cardioia.ml.predict`` (nunca treina) e grava o
resultado no contexto tipado.

Fluxo: Orquestrador -> (handoff) -> Analista de Risco -> tool do modelo -> (handoff) ->
Especialista em Protocolos.
"""

from __future__ import annotations

from typing import Any

from agents import Agent, RunContextWrapper, function_tool

from ..ml import predict
from ..schemas import risk_result
from .contexto import (
    NOME_ANALISTA_DE_RISCO,
    NOME_ESPECIALISTA_EM_PROTOCOLOS,
    NOME_TOOL_MODELO,
    ContextoDaExecucao,
)

INSTRUCOES_ANALISTA_DE_RISCO = (
    "Voce e o Analista de Risco do CardioIA (projeto academico, dados SINTETICOS).\n"
    "\n"
    "Sua unica funcao e consultar o modelo de Machine Learning ja treinado.\n"
    "\n"
    "Regras obrigatorias:\n"
    "1. Chame SEMPRE a tool 'consultar_modelo_de_risco' para obter a probabilidade. "
    "A tool nao aceita argumentos: ela usa os dados do paciente que ja estao no contexto.\n"
    "2. NUNCA estime, arredonde, 'ajuste' ou invente a probabilidade. Se a tool falhar, "
    "informe a falha exatamente como veio.\n"
    "3. NUNCA treine nem simule o modelo: a inferencia e feita em Python pela tool.\n"
    "4. Repita apenas os numeros que a tool devolver, com a mesma precisao.\n"
    "5. Depois de obter o resultado, faca o handoff para o especialista em protocolos.\n"
    "6. Este resultado NAO e diagnostico clinico e nao substitui avaliacao profissional."
)


@function_tool
def consultar_modelo_de_risco(ctx: RunContextWrapper[ContextoDaExecucao]) -> str:
    """Consulta o modelo salvo e registra a predicao no contexto (sem argumentos do LLM).

    A ausencia de parametros e proposital: se as features fossem argumentos, o LLM poderia
    alterar os valores do paciente. Aqui ele so decide **quando** consultar.
    """
    contexto = ctx.context
    if not isinstance(contexto, ContextoDaExecucao):  # pragma: no cover - defensivo
        return "FALHA: contexto de execucao ausente; a tool nao pode consultar o modelo."

    try:
        predicao = predict.prever_pico_risco(
            contexto.paciente,
            pasta_modelos=contexto.pasta_modelos,
            raiz=contexto.raiz,
            limiar=contexto.limiar,
        )
    except predict.ModeloNaoEncontradoError as erro:
        contexto.registrar(
            risk_result.TipoDePasso.ERRO,
            NOME_ANALISTA_DE_RISCO,
            f"modelo ausente: {erro}",
            sucesso=False,
        )
        return (
            "FALHA: o modelo treinado nao esta disponivel. "
            f"Detalhe: {erro} A predicao NAO foi calculada e nenhuma probabilidade deve ser "
            "informada ao usuario."
        )
    except predict.ErroDePrevisao as erro:
        contexto.registrar(
            risk_result.TipoDePasso.ERRO,
            NOME_ANALISTA_DE_RISCO,
            f"falha de inferencia: {type(erro).__name__}: {erro}",
            sucesso=False,
        )
        return (
            f"FALHA: inferencia nao concluida ({type(erro).__name__}). "
            f"Detalhe: {erro} Nao informe nenhuma probabilidade."
        )

    contexto.predicao = predicao
    contexto.registrar(
        risk_result.TipoDePasso.TOOL,
        NOME_ANALISTA_DE_RISCO,
        "predicao calculada por cardioia.ml.predict",
    )
    for aviso in predicao.get("avisos", []):
        contexto.registrar(
            risk_result.TipoDePasso.VALIDACAO, NOME_ANALISTA_DE_RISCO, f"aviso: {aviso}"
        )

    resumo = predict.resumo_para_agente(predicao)
    return (
        "Predicao validada pelo modelo de ML (valores autoritativos, nao arredondar):\n"
        f"{resumo}\n"
        f"probabilidade_risco_crua={predicao['probabilidade_risco']}\n"
        f"modelo={predicao['modelo_utilizado']} | classe_do_estimador="
        f"{predicao['classe_do_estimador']} | limiar={predicao['limiar_decisao']}"
    )


def criar_analista_de_risco(proximo_agente: Agent[Any]) -> Agent[ContextoDaExecucao]:
    """Cria o Analista de Risco, com handoff real para o Especialista em Protocolos."""
    return Agent[ContextoDaExecucao](
        name=NOME_ANALISTA_DE_RISCO,
        handoff_description=(
            "Consulta o modelo de Machine Learning treinado e devolve a probabilidade real de "
            "pico de risco. Encaminhe para ele para obter a predicao."
        ),
        instructions=INSTRUCOES_ANALISTA_DE_RISCO,
        tools=[consultar_modelo_de_risco],
        handoffs=[proximo_agente],
    )


__all__ = [
    "INSTRUCOES_ANALISTA_DE_RISCO",
    "NOME_TOOL_MODELO",
    "consultar_modelo_de_risco",
    "criar_analista_de_risco",
    "NOME_ESPECIALISTA_EM_PROTOCOLOS",
]