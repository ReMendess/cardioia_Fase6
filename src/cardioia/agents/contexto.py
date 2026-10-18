"""Contexto compartilhado do pipeline multiagente (CardioIA — Fase 6).

O SDK tem duas "camadas" de contexto: o que o **LLM vê** (histórico de mensagens) e o que
o **código lê** (``RunContextWrapper.context``). Este módulo define a segunda: um objeto
tipado, local, que **nunca é enviado ao modelo** (documentação oficial do SDK).

Por que isso importa aqui: a probabilidade e os protocolos são calculados por Python. Eles
são gravados aqui e lidos pelo orquestrador; o LLM só vê um resumo textual. Assim nenhum
número crítico depende de o modelo "lembrar" ou "repetir" corretamente.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agents import RunHooks

from ..config import MAX_TURNOS_PADRAO
from ..knowledge import protocols as camada_de_protocolos
from ..ml import predict
from ..schemas import risk_result

# --------------------------------------------------------------------------- #
# Nomes fixos do pipeline (usados nas instruções, nos testes e na trilha)
# --------------------------------------------------------------------------- #

NOME_ORQUESTRADOR = "Orquestrador"
NOME_ANALISTA_DE_RISCO = "Analista de Risco"
NOME_ESPECIALISTA_EM_PROTOCOLOS = "Especialista em Protocolos"
NOME_PLANEJADOR_DE_RESPOSTA = "Planejador de Resposta"

NOME_TOOL_MODELO = "consultar_modelo_de_risco"
NOME_TOOL_PROTOCOLOS = "consultar_protocolos"

# ``MAX_TURNOS_PADRAO`` é reexportado de ``cardioia.config`` (fonte única).


@dataclass
class PassoRegistrado:
    """Uma linha da trilha de execução (handoff, tool, validação ou erro)."""

    tipo: risk_result.TipoDePasso
    agente: str
    detalhe: str
    sucesso: bool = True

    def como_contrato(self, ordem: int) -> risk_result.PassoDaExecucao:
        return risk_result.PassoDaExecucao(
            ordem=ordem, tipo=self.tipo, agente=self.agente, detalhe=self.detalhe, sucesso=self.sucesso
        )


@dataclass
class ContextoDaExecucao:
    """Estado tipado compartilhado por tools, handoffs e agentes durante o run.

    Attributes:
        paciente: dados já validados do paciente (nomes padronizados do projeto).
        raiz: raiz do projeto (para localizar ``data/`` e ``models/``).
        pasta_modelos: pasta dos artefatos do modelo (permite injeção em testes).
        limiar: limiar de decisão usado na predição (operacional, não clínico).
        predicao: resultado **cru** de ``predict.prever_pico_risco`` (calculado por tool).
        avaliacao_de_protocolos: resultado da camada determinística de protocolos.
        passos: trilha da execução, na ordem em que aconteceu.
        erros: falhas registradas (nunca escondidas).
    """

    paciente: dict[str, float]
    raiz: Path | None = None
    pasta_modelos: Path | None = None
    limiar: float = predict.LIMIAR_DE_DECISAO
    predicao: dict[str, Any] | None = None
    avaliacao_de_protocolos: camada_de_protocolos.AvaliacaoDosProtocolos | None = None
    passos: list[PassoRegistrado] = field(default_factory=list)
    erros: list[str] = field(default_factory=list)
    handoffs: list[tuple[str, str]] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Registro
    # ------------------------------------------------------------------ #

    def registrar(
        self, tipo: risk_result.TipoDePasso, agente: str, detalhe: str, sucesso: bool = True
    ) -> None:
        """Acrescenta um passo à trilha (e, se for erro, também a ``erros``)."""
        self.passos.append(PassoRegistrado(tipo=tipo, agente=agente, detalhe=detalhe, sucesso=sucesso))
        if tipo is risk_result.TipoDePasso.ERRO or not sucesso:
            self.erros.append(f"[{agente}] {detalhe}")

    @property
    def passos_do_contrato(self) -> list[risk_result.PassoDaExecucao]:
        return [passo.como_contrato(i) for i, passo in enumerate(self.passos, start=1)]

    @property
    def handoffs_realizados(self) -> list[str]:
        return [f"{origem} -> {destino}" for origem, destino in self.handoffs]

    @property
    def tem_predicao(self) -> bool:
        return self.predicao is not None


class TrilhaDeExecucao(RunHooks[Any]):
    """``RunHooks`` do SDK que registra handoffs, tools e agentes na trilha do contexto.

    Isto dá a prova **observável** de que os handoffs aconteceram (não apenas de que os
    agentes foram instanciados): cada ``on_handoff`` é gravado no contexto.

    Os callbacks do SDK são ``async`` (ver ``RunHooksBase``): sobrescrevemos com ``async``.
    """

    async def on_agent_start(self, context: Any, agent: Any) -> None:  # noqa: D102 - do SDK
        contexto = getattr(context, "context", None)
        if isinstance(contexto, ContextoDaExecucao):
            contexto.registrar(
                risk_result.TipoDePasso.AGENTE, agent.name, f"agente '{agent.name}' iniciou o turno"
            )

    async def on_handoff(self, context: Any, from_agent: Any, to_agent: Any) -> None:  # noqa: D102
        contexto = getattr(context, "context", None)
        if isinstance(contexto, ContextoDaExecucao):
            contexto.handoffs.append((from_agent.name, to_agent.name))
            contexto.registrar(
                risk_result.TipoDePasso.HANDOFF,
                from_agent.name,
                f"handoff de '{from_agent.name}' para '{to_agent.name}'",
            )

    async def on_tool_start(self, context: Any, agent: Any, tool: Any) -> None:  # noqa: D102
        contexto = getattr(context, "context", None)
        if isinstance(contexto, ContextoDaExecucao):
            contexto.registrar(
                risk_result.TipoDePasso.TOOL,
                agent.name,
                f"tool '{getattr(tool, 'name', tool)}' chamada",
            )


__all__ = [
    "MAX_TURNOS_PADRAO",
    "NOME_ANALISTA_DE_RISCO",
    "NOME_ESPECIALISTA_EM_PROTOCOLOS",
    "NOME_ORQUESTRADOR",
    "NOME_PLANEJADOR_DE_RESPOSTA",
    "NOME_TOOL_MODELO",
    "NOME_TOOL_PROTOCOLOS",
    "ContextoDaExecucao",
    "PassoRegistrado",
    "TrilhaDeExecucao",
]