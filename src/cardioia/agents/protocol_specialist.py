"""Agente 2 — Especialista em Protocolos.

Responsabilidade única: determinar **quais protocolos oficiais** se aplicam, usando a
camada determinística ``cardioia.knowledge.protocols``. O LLM não cria regra, não altera
limiar e não recomenda protocolo fora da base da aula.

A tool exige que a predição do Analista de Risco já esteja no contexto: uma das regras
oficiais ("risco sistêmico alto") depende da classe prevista pelo modelo de ML.

Fluxo: Analista de Risco -> (handoff) -> Especialista em Protocolos -> tool de protocolos
-> (handoff) -> Planejador de Resposta.
"""

from __future__ import annotations

from typing import Any

from agents import Agent, RunContextWrapper, function_tool

from ..knowledge import protocols as camada_de_protocolos
from ..schemas import risk_result
from .contexto import (
    NOME_ESPECIALISTA_EM_PROTOCOLOS,
    NOME_PLANEJADOR_DE_RESPOSTA,
    NOME_TOOL_PROTOCOLOS,
    ContextoDaExecucao,
)

INSTRUCOES_ESPECIALISTA_EM_PROTOCOLOS = (
    "Voce e o Especialista em Protocolos do CardioIA (projeto academico, dados SINTETICOS).\n"
    "\n"
    "Sua unica funcao e identificar quais protocolos oficiais da base de conhecimento da "
    "aula se aplicam ao paciente.\n"
    "\n"
    "Regras obrigatorias:\n"
    "1. Chame SEMPRE a tool 'consultar_protocolos'. Ela nao aceita argumentos: le o paciente "
    "e a predicao validada que ja estao no contexto.\n"
    "2. NUNCA invente protocolo, NUNCA altere limites e NUNCA sugira conduta que nao esteja "
    "no texto devolvido pela tool.\n"
    "3. Repita os protocolos exatamente com o texto oficial e o valor observado que a tool "
    "informar.\n"
    "4. Se a tool falhar (por exemplo, predicao ausente), informe a falha tal como veio.\n"
    "5. Depois de obter o resultado, faca o handoff para o planejador de resposta.\n"
    "6. Os limites sao de PROJETO (materiais da aula) e nao configuram conduta clinica."
)


@function_tool
def consultar_protocolos(ctx: RunContextWrapper[ContextoDaExecucao]) -> str:
    """Aplica as 10 regras oficiais ao paciente e registra o resultado no contexto."""
    contexto = ctx.context
    if not isinstance(contexto, ContextoDaExecucao):  # pragma: no cover - defensivo
        return "FALHA: contexto de execucao ausente; a tool nao pode avaliar protocolos."

    if not contexto.tem_predicao:
        contexto.registrar(
            risk_result.TipoDePasso.ERRO,
            NOME_ESPECIALISTA_EM_PROTOCOLOS,
            "predicao ausente no contexto: a regra 'risco sistemico alto' depende dela",
            sucesso=False,
        )
        return (
            "FALHA: a predicao do modelo de ML ainda nao foi calculada, e a regra de risco "
            "sistemico depende dela. Nao e possivel avaliar os protocolos agora."
        )

    assert contexto.predicao is not None  # garantido por tem_predicao
    classe_prevista = int(contexto.predicao["pico_risco_previsto"])

    avaliacao, erros = camada_de_protocolos.avaliar_protocolos_seguro(
        contexto.paciente, classe_prevista, raiz=contexto.raiz
    )
    if avaliacao is None:
        contexto.registrar(
            risk_result.TipoDePasso.ERRO,
            NOME_ESPECIALISTA_EM_PROTOCOLOS,
            f"falha ao avaliar protocolos: {'; '.join(erros)}",
            sucesso=False,
        )
        return (
            "FALHA: nao foi possivel avaliar a base de protocolos. Detalhe: "
            f"{'; '.join(erros)} Nenhum protocolo deve ser sugerido."
        )

    contexto.avaliacao_de_protocolos = avaliacao
    contexto.registrar(
        risk_result.TipoDePasso.TOOL,
        NOME_ESPECIALISTA_EM_PROTOCOLOS,
        f"protocolos avaliados pela camada deterministica ({avaliacao.quantidade} aplicaveis)",
    )
    for aviso in avaliacao.avisos:
        contexto.registrar(
            risk_result.TipoDePasso.VALIDACAO, NOME_ESPECIALISTA_EM_PROTOCOLOS, f"aviso: {aviso}"
        )

    return (
        "Protocolos oficiais aplicaveis (texto e valores autoritativos, nao reescrever):\n"
        f"{avaliacao.resumo()}\n"
        f"total_avaliado={avaliacao.total_avaliado} | aplicaveis={avaliacao.quantidade}"
    )


def criar_especialista_em_protocolos(proximo_agente: Agent[Any]) -> Agent[ContextoDaExecucao]:
    """Cria o Especialista em Protocolos, com handoff real para o Planejador de Resposta."""
    return Agent[ContextoDaExecucao](
        name=NOME_ESPECIALISTA_EM_PROTOCOLOS,
        handoff_description=(
            "Aplica as regras deterministicas da base de conhecimento oficial e devolve "
            "somente os protocolos aplicaveis, com o valor observado que os justifica."
        ),
        instructions=INSTRUCOES_ESPECIALISTA_EM_PROTOCOLOS,
        tools=[consultar_protocolos],
        handoffs=[proximo_agente],
    )


__all__ = [
    "INSTRUCOES_ESPECIALISTA_EM_PROTOCOLOS",
    "NOME_PLANEJADOR_DE_RESPOSTA",
    "NOME_TOOL_PROTOCOLOS",
    "consultar_protocolos",
    "criar_especialista_em_protocolos",
]