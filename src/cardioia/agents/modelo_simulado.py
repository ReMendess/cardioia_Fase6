"""Modelo simulado para executar o pipeline **sem a API da OpenAI** (testes e demonstração).

Por que existe: o ambiente pode não ter ``OPENAI_API_KEY``. Em vez de declarar o pipeline
"validado" sem prova, injetamos um ``Model`` do SDK que executa um **roteiro** — e deixamos
todo o resto do pipeline real: os ``Agent``, os *handoffs* do SDK, as *function tools* do SDK,
o ``RunHooks`` e a validação do contrato.

O que este modelo **não** faz: ele não decide nada por conta própria. Ele valida cada passo
do roteiro contra as tools/handoffs que o SDK realmente ofereceu naquele turno; se o passo
for impossível, registra o erro (e o pipeline termina como ``ERRO``, não como sucesso).

Os nomes das tools de handoff são resolvidos pelo campo ``agent_name`` do próprio ``Handoff``
(não há nome hardcoded): assim a implementação acompanha a convenção do SDK.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from agents.handoffs import Handoff
from agents.items import ModelResponse
from agents.models.interface import Model, ModelProvider, ModelTracing
from agents.usage import Usage
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)

TipoDePassoDoRoteiro = Literal["handoff", "tool", "final"]


@dataclass(frozen=True)
class PassoDoRoteiro:
    """Um passo esperado do pipeline: handoff para alguém, chamada de tool ou saída final."""

    tipo: TipoDePassoDoRoteiro
    alvo: str
    """Nome do agente (handoff), nome da tool (tool) ou texto JSON (final)."""


def _mensagem(texto: str, identificador: str) -> ResponseOutputMessage:
    return ResponseOutputMessage(
        id=identificador,
        type="message",
        role="assistant",
        status="completed",
        content=[ResponseOutputText(type="output_text", text=texto, annotations=[])],
    )


def _chamada_de_funcao(nome: str, identificador: str) -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall(
        id=f"fc_{identificador}",
        type="function_call",
        name=nome,
        arguments="{}",
        call_id=f"call_{identificador}",
        status="completed",
    )


@dataclass
class ModeloComRoteiro(Model):
    """``Model`` do SDK que executa um roteiro fixo, validando cada passo.

    Attributes:
        roteiro: fila de passos. Consome exatamente um passo por chamada ao modelo — é o que
            impede o laço infinito de um modelo que "repetiria" a mesma tool para sempre.
        chamadas: registro textual de cada turno (tools/handoffs vistos).
        problemas: inconsistências entre o roteiro e o que o SDK ofereceu.
    """

    roteiro: list[PassoDoRoteiro] = field(default_factory=list)
    chamadas: list[str] = field(default_factory=list)
    problemas: list[str] = field(default_factory=list)
    historico: list[list[str]] = field(default_factory=list)
    """Tipos dos itens recebidos em cada turno (evidência do histórico do SDK)."""

    @staticmethod
    def _tipos_dos_itens(entrada: Any) -> list[str]:
        """Tipos dos itens de entrada do turno (``message``, ``function_call``, ...)."""
        if not isinstance(entrada, list):
            return [type(entrada).__name__]
        tipos: list[str] = []
        for item in entrada:
            if isinstance(item, dict):
                tipos.append(str(item.get("type", "message")))
            else:
                tipos.append(str(getattr(item, "type", type(item).__name__)))
        return tipos

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[Any],
        model_settings: Any,
        tools: list[Any],
        output_schema: Any,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: Any | None,
    ) -> ModelResponse:
        """Devolve o próximo passo do roteiro, conferindo-o contra tools/handoffs recebidos."""
        nomes_tools = [getattr(tool, "name", str(tool)) for tool in tools]
        nomes_agentes = [getattr(h, "agent_name", getattr(h, "tool_name", "?")) for h in handoffs]
        numero = len(self.chamadas) + 1
        self.historico.append(self._tipos_dos_itens(input))
        self.chamadas.append(
            f"turno {numero}: tools={nomes_tools} handoffs={nomes_agentes} "
            f"output_schema={'sim' if output_schema is not None else 'nao'} "
            f"itens_no_historico={self.historico[-1]}"
        )

        if not self.roteiro:
            self.problemas.append(f"turno {numero}: roteiro esgotado antes do fim do run")
            return self._resposta(_mensagem("ROTEIRO ESGOTADO", f"msg_{numero}"), numero)

        passo = self.roteiro.pop(0)

        if passo.tipo == "tool":
            if passo.alvo not in nomes_tools:
                self.problemas.append(
                    f"turno {numero}: tool '{passo.alvo}' indisponivel (disponiveis: {nomes_tools})"
                )
            return self._resposta(_chamada_de_funcao(passo.alvo, f"t{numero}"), numero)

        if passo.tipo == "handoff":
            oferecido = next(
                (h for h in handoffs if getattr(h, "agent_name", None) == passo.alvo), None
            )
            if oferecido is None:
                self.problemas.append(
                    f"turno {numero}: handoff para '{passo.alvo}' indisponivel "
                    f"(disponiveis: {nomes_agentes})"
                )
                return self._resposta(_mensagem("HANDOFF INDISPONIVEL", f"msg_{numero}"), numero)
            return self._resposta(_chamada_de_funcao(oferecido.tool_name, f"h{numero}"), numero)

        return self._resposta(_mensagem(passo.alvo, f"m{numero}"), numero)

    @staticmethod
    def _resposta(saida: Any, numero: int) -> ModelResponse:
        """Monta a resposta do turno. Aceita um item ou uma lista (normaliza aqui).

        Normalizar neste ponto evita um erro sutil: ``output`` é ``list[...]`` e, ao receber
        um objeto único, o pydantic tentaria iterá-lo — e os ``BaseModel`` do pacote
        ``openai`` implementam ``__iter__``, o que produziria um erro enganoso de validação.
        """
        itens = saida if isinstance(saida, list) else [saida]
        return ModelResponse(
            output=itens,
            usage=Usage(),
            response_id=f"resp_simulado_{numero}",
            request_id=f"req_simulado_{numero}",
        )

    def stream_response(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError("O pipeline do CardioIA usa Runner.run (nao streaming).")

    @property
    def resumo_das_chamadas(self) -> str:
        return "\n".join(self.chamadas)


@dataclass
class ProvedorDeModelosSimulado(ModelProvider):
    """``ModelProvider`` que devolve sempre o mesmo modelo simulado."""

    modelo: Model

    def get_model(self, model_name: str | None) -> Model:  # noqa: ARG002 - assinatura do SDK
        return self.modelo

    async def aclose(self) -> None:  # pragma: no cover - sem recursos a liberar
        return None


__all__ = [
    "ModeloComRoteiro",
    "PassoDoRoteiro",
    "ProvedorDeModelosSimulado",
    "TipoDePassoDoRoteiro",
]