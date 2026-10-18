"""Agente 3 — Planejador de Resposta.

Responsabilidade única: **consolidar** a resposta final em texto, usando exclusivamente os
resultados que já estão no histórico (o resumo da predição e a lista de protocolos). Ele não
recalcula a probabilidade, não altera os protocolos e não cria números.

A saída é **estruturada** (``output_type`` = :class:`ContribuicaoDoPlanejador`), o que faz o
SDK validar o formato. Os valores numéricos autoritativos **não** vêm daqui: eles são lidos
do contexto pelo orquestrador. Se a narrativa citar um número divergente, o orquestrador
marca ``CONCLUIDO_COM_RESSALVAS``.

Fluxo: Especialista em Protocolos -> (handoff) -> Planejador de Resposta -> saída final.
"""

from __future__ import annotations

from typing import Any

from agents import Agent
from pydantic import BaseModel, Field

from .contexto import NOME_PLANEJADOR_DE_RESPOSTA, ContextoDaExecucao

INSTRUCOES_PLANEJADOR_DE_RESPOSTA = (
    "Voce e o Planejador de Resposta do CardioIA (projeto academico, dados SINTETICOS).\n"
    "\n"
    "Sua funcao e consolidar, em linguagem clara e nao alarmista, o que ja foi calculado "
    "pelas outras etapas do pipeline.\n"
    "\n"
    "Regras obrigatorias:\n"
    "1. Use SOMENTE os valores que aparecem no historico desta conversa (resumo da predicao "
    "e lista de protocolos). NAO recalcule, NAO estime e NAO crie numeros.\n"
    "2. Se precisar citar a probabilidade, copie exatamente o valor informado pela tool do "
    "modelo (mesma quantidade de casas decimais).\n"
    "3. Nao altere, renomeie nem remova protocolos: cite o texto oficial recebido.\n"
    "4. Cada protocolo citado deve vir acompanhado da justificativa objetiva (condicao "
    "satisfeita e valor observado) informada pela tool de protocolos.\n"
    "5. Deixe explicito que os dados sao sinteticos e que NAO ha validacao clinica: isto e "
    "um prototipo academico, nao diagnostico nem prescricao.\n"
    "6. Preencha tambem 'pontos_de_atencao' com limitacoes relevantes do resultado."
)


class ContribuicaoDoPlanejador(BaseModel):
    """Saída estruturada do Planejador de Resposta (validada pelo SDK via ``output_type``).

    Só contém **texto**: nenhum número autoritativo nasce aqui.
    """

    model_config = {"extra": "forbid"}

    narrativa: str = Field(
        min_length=1,
        description="Resumo em linguagem clara do que foi calculado, sem numeros novos.",
    )
    justificativas: list[str] = Field(
        default_factory=list,
        description="Uma justificativa objetiva por protocolo aplicavel (condicao + valor).",
    )
    pontos_de_atencao: list[str] = Field(
        default_factory=list,
        description="Avisos/limitacoes relevantes (dados sinteticos, sem validacao clinica).",
    )


def criar_planejador_de_resposta() -> Agent[ContextoDaExecucao]:
    """Cria o Planejador de Resposta (último agente do fluxo: sem handoffs de saída)."""
    return Agent[ContextoDaExecucao](
        name=NOME_PLANEJADOR_DE_RESPOSTA,
        handoff_description=(
            "Consolida a resposta final em linguagem clara a partir da predicao validada e dos "
            "protocolos aplicaveis. E o ultimo agente do fluxo."
        ),
        instructions=INSTRUCOES_PLANEJADOR_DE_RESPOSTA,
        output_type=ContribuicaoDoPlanejador,
    )


def narrativa_como_texto(saida: Any) -> tuple[str | None, list[str], list[str]]:
    """Extrai ``(narrativa, justificativas, pontos_de_atencao)`` do ``final_output``.

    Aceita a instância Pydantic devolvida pelo SDK, um ``dict`` equivalente, ou texto puro
    (quando o último agente não produziu saída estruturada). Nunca inventa conteúdo.
    """
    if isinstance(saida, ContribuicaoDoPlanejador):
        return saida.narrativa, list(saida.justificativas), list(saida.pontos_de_atencao)
    if isinstance(saida, dict):
        return (
            saida.get("narrativa"),
            list(saida.get("justificativas") or []),
            list(saida.get("pontos_de_atencao") or []),
        )
    if isinstance(saida, str) and saida.strip():
        return saida.strip(), [], []
    return None, [], []


__all__ = [
    "INSTRUCOES_PLANEJADOR_DE_RESPOSTA",
    "ContribuicaoDoPlanejador",
    "criar_planejador_de_resposta",
    "narrativa_como_texto",
]