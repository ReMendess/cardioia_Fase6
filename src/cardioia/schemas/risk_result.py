"""Contratos de entrada e saída do pipeline multiagente (CardioIA — Fase 6).

Modelos Pydantic usados por **todas** as camadas (agentes, orquestrador, CLI e testes):
não existem estruturas equivalentes duplicadas em outro lugar.

Regras do contrato:

- a **probabilidade** é o valor cru devolvido pelo modelo de ML (``ge=0, le=1``); o
  arredondamento acontece apenas na apresentação;
- a **classe prevista** segue a convenção de ``cardioia.ml.predict``
  (``0`` = sem pico, ``1`` = pico) e precisa ser coerente com a classificação textual;
- **falha não vira sucesso**: ``status=CONCLUIDO`` exige diagnóstico e proíbe erros
  registrados; ``INCOMPLETO``/``ERRO`` exigem ao menos um erro descrito;
- ``extra="forbid"`` em todos os modelos: campos desconhecidos são erro, não "ignorados".
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from numbers import Integral
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..ml import predict

#: Classe positiva e rótulos textuais — reaproveitados do módulo de ML (fonte única).
CLASSE_POSITIVA = predict.CLASSE_POSITIVA
ROTULOS: dict[int, str] = dict(predict.ROTULOS)

AVISO_DE_USO = predict.AVISO_DE_USO

LIMITACOES_PADRAO: tuple[str, ...] = (
    "Dados SINTETICOS gerados por regras (np.random.seed(42)); nenhum paciente real.",
    "O rotulo pico_risco e uma funcao deterministica de 6 variaveis: metricas altas "
    "refletem a reconstrucao dessa regra, nao capacidade preditiva real.",
    "Os protocolos tem limites de PROJETO (material da aula) e nao constituem conduta "
    "clinica validada.",
    "O limiar de decisao e operacional (padrao 0.5) e nao foi calibrado com dados reais.",
    "Este resultado nao e diagnostico, nao e prescricao e nao substitui avaliacao "
    "profissional de saude.",
)


class StatusDaExecucao(str, Enum):
    """Estado real do pipeline — nunca "sucesso" por omissão."""

    CONCLUIDO = "CONCLUIDO"
    CONCLUIDO_COM_RESSALVAS = "CONCLUIDO_COM_RESSALVAS"
    INCOMPLETO = "INCOMPLETO"
    ERRO = "ERRO"


class TipoDePasso(str, Enum):
    """Natureza de cada passo registrado na trilha de execução."""

    HANDOFF = "handoff"
    TOOL = "tool"
    AGENTE = "agente"
    VALIDACAO = "validacao"
    ERRO = "erro"


class ModoDeExecucao(str, Enum):
    """Como o pipeline foi executado (transparência sobre o que foi testado)."""

    SIMULADO = "simulado"
    REAL = "real"


class ModeloDoContrato(BaseModel):
    """Base comum: proíbe campos desconhecidos em todo o contrato."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class IndicadorDeterministico(ModeloDoContrato):
    """Indicador calculado em Python (nunca pelo LLM) a partir dos dados do paciente."""

    nome: str = Field(min_length=1, description="Ex.: 'carga_sistema > 80'.")
    coluna: str | None = Field(default=None, description="Campo do paciente; None se do modelo.")
    operador: Literal["<", ">", "=="]
    limiar: float
    valor_observado: float
    satisfeito: bool
    origem: str = Field(min_length=1, description="De onde veio o valor (paciente ou modelo).")


class ProtocoloAplicavel(ModeloDoContrato):
    """Protocolo oficial acionado + a justificativa objetiva (condição e valor)."""

    ordem: int = Field(ge=1, le=10, description="Posição no arquivo oficial da aula.")
    identificador: str = Field(min_length=1)
    texto_oficial: str = Field(min_length=1)
    acao: str = Field(min_length=1)
    condicao_satisfeita: str = Field(min_length=1, description="Ex.: 'carga_sistema > 85'.")
    operador: Literal["<", ">", "=="]
    limiar: float
    valor_observado: float
    origem_do_valor: str = Field(min_length=1)
    usa_modelo: bool = Field(
        default=False, description="True apenas para 'risco sistêmico alto'."
    )


class DiagnosticoDaPredicao(ModeloDoContrato):
    """Resultado validado do modelo de ML (valores vindos da tool, não do LLM)."""

    probabilidade_risco: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Valor CRU do classificador (0 a 1)."
    )
    pico_risco_previsto: Literal[0, 1]
    classificacao_risco: Literal["sem_pico_de_risco", "pico_de_risco"]
    modelo_utilizado: str = Field(min_length=1)
    classe_do_estimador: str = Field(min_length=1)
    limiar_decisao: float = Field(ge=0.0, le=1.0)
    regra_de_decisao: str = Field(min_length=1)
    features_utilizadas: list[str] = Field(min_length=1)
    origem_do_modelo: dict[str, Any] = Field(default_factory=dict)
    avisos: list[str] = Field(default_factory=list)

    @field_validator("pico_risco_previsto", mode="before")
    @classmethod
    def _classe_estritamente_inteira(cls, valor: Any) -> Any:
        """Rejeita ``bool`` e tipos não inteiros na classe prevista.

        Sem isso, o pydantic (modo não estrito de ``Literal[0, 1]``) converteria
        ``True`` em ``1`` — e um booleano passaria como "pico de risco". A convenção do
        módulo de ML usa inteiros; coerção silenciosa aqui seria uma fonte de confusão.
        """
        if isinstance(valor, bool) or not isinstance(valor, Integral):
            raise ValueError(
                "'pico_risco_previsto' deve ser o inteiro 0 ou 1 (sem booleano, sem texto); "
                f"recebido {valor!r} ({type(valor).__name__})."
            )
        return int(valor)

    @model_validator(mode="after")
    def _classe_coerente_com_classificacao(self) -> DiagnosticoDaPredicao:
        esperado = ROTULOS[int(self.pico_risco_previsto)]
        if self.classificacao_risco != esperado:
            raise ValueError(
                f"classificacao_risco '{self.classificacao_risco}' nao corresponde a "
                f"pico_risco_previsto={self.pico_risco_previsto} (esperado '{esperado}')."
            )
        return self


class PassoDaExecucao(ModeloDoContrato):
    """Passo da trilha de execução (handoff, tool, validação ou erro)."""

    ordem: int = Field(ge=1)
    tipo: TipoDePasso
    agente: str = Field(min_length=1)
    detalhe: str = Field(min_length=1)
    sucesso: bool = True


class RespostaDaExecucao(ModeloDoContrato):
    """Resposta final estruturada do pipeline multiagente.

    É o **contrato único** de saída: o orquestrador monta, a CLI imprime/serializa e os
    testes verificam. Nenhuma estrutura paralela é criada em outro módulo.
    """

    id_execucao: str = Field(min_length=8, description="Identificador da execução (UUID4).")
    gerado_em: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Timestamp UTC."
    )
    status: StatusDaExecucao
    modo: ModoDeExecucao = ModoDeExecucao.SIMULADO
    diagnostico: DiagnosticoDaPredicao | None = None
    indicadores: list[IndicadorDeterministico] = Field(default_factory=list)
    protocolos: list[ProtocoloAplicavel] = Field(default_factory=list)
    justificativas: list[str] = Field(default_factory=list)
    narrativa: str | None = Field(
        default=None, description="Texto do Planejador de Resposta (não contém números novos)."
    )
    avisos: list[str] = Field(default_factory=list)
    limitacoes: list[str] = Field(default_factory=lambda: list(LIMITACOES_PADRAO))
    erros: list[str] = Field(default_factory=list)
    passos: list[PassoDaExecucao] = Field(default_factory=list)
    handoffs_realizados: list[str] = Field(default_factory=list)
    agente_final: str | None = None
    versao_do_sdk: str | None = None
    duracao_segundos: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def _coerencia_entre_status_e_conteudo(self) -> RespostaDaExecucao:
        """Impede respostas "bem-sucedidas" com falha escondida (e vice-versa)."""
        if self.status is StatusDaExecucao.CONCLUIDO:
            if self.diagnostico is None:
                raise ValueError(
                    "status=CONCLUIDO exige 'diagnostico'; sem predicao validada a "
                    "execucao nao pode ser considerada concluida."
                )
            if self.erros:
                raise ValueError(
                    "status=CONCLUIDO nao pode conter 'erros': use "
                    "CONCLUIDO_COM_RESSALVAS ou ERRO."
                )
        if self.status in (StatusDaExecucao.ERRO, StatusDaExecucao.INCOMPLETO):
            if not self.erros:
                raise ValueError(
                    f"status={self.status.value} exige ao menos um item em 'erros' "
                    "(nada de falha silenciosa)."
                )
        return self

    @property
    def concluida(self) -> bool:
        """``True`` apenas quando o pipeline terminou com predição validada."""
        return self.status in (
            StatusDaExecucao.CONCLUIDO,
            StatusDaExecucao.CONCLUIDO_COM_RESSALVAS,
        )

    def como_dicionario(self) -> dict[str, Any]:
        """Serializa em tipos nativos (JSON-safe), preservando a probabilidade crua."""
        return self.model_dump(mode="json")

    def resumo_textual(self) -> str:
        """Resumo legível para a CLI (aqui sim a probabilidade é arredondada)."""
        linhas = [
            "=" * 78,
            "CardioIA - Fase 6 | Resposta do pipeline multiagente",
            "=" * 78,
            f"  id da execucao : {self.id_execucao}",
            f"  gerado em      : {self.gerado_em.isoformat()}",
            f"  status         : {self.status.value}",
            f"  modo           : {self.modo.value}",
            f"  agente final   : {self.agente_final or '-'}",
        ]
        if self.diagnostico:
            diagnostico = self.diagnostico
            probabilidade = diagnostico.probabilidade_risco
            texto = "indisponivel" if probabilidade is None else f"{probabilidade:.4f}"
            linhas += [
                "",
                "  PREDICAO (calculada pelo modelo de ML, nunca pelo LLM)",
                f"    modelo                 : {diagnostico.modelo_utilizado} "
                f"({diagnostico.classe_do_estimador})",
                f"    probabilidade de pico  : {texto}",
                f"    pico_risco_previsto    : {diagnostico.pico_risco_previsto}",
                f"    classificacao de risco : {diagnostico.classificacao_risco}",
                f"    limiar de decisao      : {diagnostico.limiar_decisao} "
                f"({diagnostico.regra_de_decisao})",
            ]
        linhas += ["", f"  PROTOCOLOS APLICAVEIS ({len(self.protocolos)})"]
        if self.protocolos:
            for protocolo in self.protocolos:
                linhas.append(f"    - {protocolo.texto_oficial}")
                linhas.append(
                    f"        condicao: {protocolo.condicao_satisfeita} | "
                    f"valor observado: {protocolo.valor_observado:g} | "
                    f"origem: {protocolo.origem_do_valor}"
                )
        else:
            linhas.append("    (nenhum protocolo oficial foi acionado)")
        if self.justificativas:
            linhas += ["", "  JUSTIFICATIVAS"] + [f"    - {item}" for item in self.justificativas]
        if self.narrativa:
            linhas += ["", "  NARRATIVA (Planejador de Resposta)", f"    {self.narrativa}"]
        if self.handoffs_realizados:
            linhas += ["", "  HANDOFFS REALIZADOS"] + [
                f"    -> {item}" for item in self.handoffs_realizados
            ]
        if self.passos:
            linhas += ["", "  TRILHA DE EXECUCAO"]
            for passo in self.passos:
                marca = "ok" if passo.sucesso else "falha"
                linhas.append(
                    f"    {passo.ordem}. [{passo.tipo.value}/{marca}] {passo.agente}: {passo.detalhe}"
                )
        if self.erros:
            linhas += ["", "  ERROS"] + [f"    - {item}" for item in self.erros]
        if self.avisos:
            linhas += ["", "  AVISOS"] + [f"    - {item}" for item in self.avisos]
        linhas += ["", "  LIMITACOES"] + [f"    - {item}" for item in self.limitacoes]
        return "\n".join(linhas)


def contrato_de_exemplo() -> dict[str, Any]:
    """Exemplo mínimo válido do contrato (usado em testes e na documentação)."""
    return {
        "id_execucao": "00000000-0000-4000-8000-000000000000",
        "status": StatusDaExecucao.INCOMPLETO.value,
        "erros": ["exemplo ilustrativo: nenhum pipeline foi executado"],
    }


__all__ = [
    "AVISO_DE_USO",
    "CLASSE_POSITIVA",
    "LIMITACOES_PADRAO",
    "ROTULOS",
    "DiagnosticoDaPredicao",
    "IndicadorDeterministico",
    "ModoDeExecucao",
    "PassoDaExecucao",
    "ProtocoloAplicavel",
    "RespostaDaExecucao",
    "StatusDaExecucao",
    "TipoDePasso",
    "contrato_de_exemplo",
]