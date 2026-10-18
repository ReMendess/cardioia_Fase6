"""Agente 4 — Orquestrador (e ponto de entrada do pipeline).

Responsabilidades:

1. **Validar a entrada** do paciente (antes de qualquer chamada de modelo).
2. **Verificar se o modelo treinado existe** — nunca treinar silenciosamente.
3. Iniciar o fluxo e encaminhar ao **Analista de Risco** (handoff real do SDK).
4. Garantir que a predição validada chegue ao **Especialista em Protocolos** (via contexto).
5. Garantir que predição + protocolos cheguem ao **Planejador de Resposta**.
6. Montar e **validar** a resposta final no contrato Pydantic
   (:class:`cardioia.schemas.risk_result.RespostaDaExecucao`).
7. Registrar status e erros da execução — sucesso **nunca** por omissão.

**Adaptação exigida pelo SDK (documentada):** handoffs são unidirecionais dentro de um único
``Runner.run`` e o controle **não volta** ao orquestrador; quem produz a saída final é o último
agente. Por isso o orquestrador dispara o ``run`` com um **contexto tipado** compartilhado, e a
montagem do contrato acontece em Python depois do run, lendo os valores autoritativos do
contexto (não do texto do LLM).
"""

from __future__ import annotations

import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agents import Agent, RunConfig, Runner
from agents.exceptions import AgentsException

from ..config import VARIAVEL_DE_CHAVE
from ..ml import predict
from ..schemas import risk_result
from . import modelo_simulado as simulacao
from .contexto import (
    MAX_TURNOS_PADRAO,
    NOME_ANALISTA_DE_RISCO,
    NOME_ESPECIALISTA_EM_PROTOCOLOS,
    NOME_ORQUESTRADOR,
    NOME_PLANEJADOR_DE_RESPOSTA,
    ContextoDaExecucao,
    TrilhaDeExecucao,
)
from .protocol_specialist import criar_especialista_em_protocolos
from .response_planner import ContribuicaoDoPlanejador, criar_planejador_de_resposta, narrativa_como_texto
from .risk_analyst import criar_analista_de_risco

INSTRUCOES_ORQUESTRADOR = (
    "Voce e o Orquestrador do CardioIA (projeto academico, dados SINTETICOS).\n"
    "\n"
    "Coordene o pipeline na ordem exata:\n"
    "1. Faca o handoff para o Analista de Risco (ele consulta o modelo de ML).\n"
    "2. Depois do Analista, o fluxo segue para o Especialista em Protocolos.\n"
    "3. Depois do Especialista, o fluxo segue para o Planejador de Resposta, que consolida "
    "a resposta final.\n"
    "\n"
    "Regras obrigatorias:\n"
    "- Nao responda ao usuario por conta propria: sempre encaminhe pelo handoff.\n"
    "- Nao calcule nem estime probabilidade, protocolo ou limiar.\n"
    "- Nao altere nem reinterprete os resultados das tools.\n"
    "- Se alguma tool falhar, relate a falha exatamente como veio."
)

ENTRADA_PADRAO = (
    "Avalie o paciente fornecido usando o fluxo completo do CardioIA: consulta ao modelo de "
    "Machine Learning, avaliacao dos protocolos oficiais e consolidacao da resposta final. "
    "Os dados do paciente ja estao no contexto da execucao."
)

AVISO_MODO_SIMULADO = (
    "Execucao em modo SIMULADO: as decisoes de rota do LLM foram substituidas por um roteiro "
    "deterministico (o SDK, os handoffs e as tools sao os reais). Nao equivale a uma chamada "
    "real a API da OpenAI."
)

_ERRO_DE_CHAVE = (
    "OPENAI_API_KEY nao configurada. O modo real precisa da chave em variavel de ambiente "
    "(Windows: $env:OPENAI_API_KEY = 'sk-...'). Alternativa sem chave: --modo simulado."
)


@dataclass
class AgentesDoPipeline:
    """Os quatro agentes, já ligados pelos handoffs reais do SDK."""

    orquestrador: Agent[ContextoDaExecucao]
    analista_de_risco: Agent[ContextoDaExecucao]
    especialista_em_protocolos: Agent[ContextoDaExecucao]
    planejador_de_resposta: Agent[ContextoDaExecucao]

    def nomes(self) -> list[str]:
        return [
            self.orquestrador.name,
            self.analista_de_risco.name,
            self.especialista_em_protocolos.name,
            self.planejador_de_resposta.name,
        ]


def construir_agentes() -> AgentesDoPipeline:
    """Instancia os quatro agentes com handoffs reais (Analista -> Especialista -> Planejador)."""
    planejador = criar_planejador_de_resposta()
    especialista = criar_especialista_em_protocolos(planejador)
    analista = criar_analista_de_risco(especialista)
    orquestrador = Agent[ContextoDaExecucao](
        name=NOME_ORQUESTRADOR,
        handoff_description="Coordena o pipeline completo do CardioIA.",
        instructions=INSTRUCOES_ORQUESTRADOR,
        handoffs=[analista],
    )
    return AgentesDoPipeline(
        orquestrador=orquestrador,
        analista_de_risco=analista,
        especialista_em_protocolos=especialista,
        planejador_de_resposta=planejador,
    )


def roteiro_padrao() -> list[simulacao.PassoDoRoteiro]:
    """Roteiro determinístico do fluxo completo (usado no modo simulado).

    São **seis** turnos: os cinco passos do fluxo (handoff, tool, handoff, tool, handoff) mais
    o turno final do Planejador de Resposta, que produz a saída estruturada.
    """
    return [
        simulacao.PassoDoRoteiro("handoff", NOME_ANALISTA_DE_RISCO),
        simulacao.PassoDoRoteiro("tool", "consultar_modelo_de_risco"),
        simulacao.PassoDoRoteiro("handoff", NOME_ESPECIALISTA_EM_PROTOCOLOS),
        simulacao.PassoDoRoteiro("tool", "consultar_protocolos"),
        simulacao.PassoDoRoteiro("handoff", NOME_PLANEJADOR_DE_RESPOSTA),
        simulacao.PassoDoRoteiro("final", TEXTO_FINAL_SIMULADO),
    ]


#: Narrativa devolvida pelo Planejador no **modo simulado**.
#: Não contém nenhum número: nenhum valor pode nascer do texto (eles vêm das tools e do
#: contexto). Serve apenas para exercitar a saída estruturada (``output_type``) do SDK.
TEXTO_FINAL_SIMULADO = (
    '{"narrativa": "Execucao em modo simulado. A probabilidade, a classe prevista e os '
    'protocolos aplicaveis foram calculados pelas tools em Python e estao registrados na '
    'trilha de execucao; consulte os campos estruturados da resposta para os valores '
    'autoritativos.", '
    '"justificativas": [], '
    '"pontos_de_atencao": ["Narrativa fixa do modo simulado: nenhum modelo de linguagem foi '
    'chamado nesta execucao."]}\n'
)


# --------------------------------------------------------------------------- #
# Montagem do contrato final
# --------------------------------------------------------------------------- #


def _indicadores_do_paciente(paciente: dict[str, float]) -> list[risk_result.IndicadorDeterministico]:
    """Indicadores do rótulo oficial calculados em Python (nunca pelo LLM).

    Usa a tabela canônica de ``cardioia.ml.dataset`` (mesma que gera o rótulo) em vez de
    duplicar limites aqui — a duplicação seria uma fonte de divergência silenciosa. O
    operador é lido do próprio texto do indicador (ex.: ``"carga_sistema > 80"``) e a
    coerência com ``dataset.indicadores_ativos`` é verificada a cada montagem.
    """
    from ..ml import dataset as camada_de_dados

    indicadores: list[risk_result.IndicadorDeterministico] = []
    satisfeitos_calculados: set[str] = set()
    for rotulo, chave, comparador, limiar in camada_de_dados._INDICADORES:  # noqa: SLF001
        if chave not in paciente:
            continue
        valor = float(paciente[chave])
        satisfeito = bool(comparador(valor, float(limiar)))
        if satisfeito:
            satisfeitos_calculados.add(rotulo)
        indicadores.append(
            risk_result.IndicadorDeterministico(
                nome=rotulo,
                coluna=chave,
                operador="<" if "<" in rotulo else ">",
                limiar=float(limiar),
                valor_observado=valor,
                satisfeito=satisfeito,
                origem="dados_do_paciente",
            )
        )

    satisfeitos_oficiais = set(camada_de_dados.indicadores_ativos(paciente))
    if satisfeitos_calculados != satisfeitos_oficiais:  # pragma: no cover - guarda de drift
        raise RuntimeError(
            "Divergencia entre os indicadores montados e os da base oficial: "
            f"{satisfeitos_calculados ^ satisfeitos_oficiais}"
        )
    return indicadores


def _protocolos_do_contrato(
    contexto: ContextoDaExecucao,
) -> list[risk_result.ProtocoloAplicavel]:
    if contexto.avaliacao_de_protocolos is None:
        return []
    return [
        risk_result.ProtocoloAplicavel(
            ordem=aplicado.ordem,
            identificador=aplicado.identificador,
            texto_oficial=aplicado.texto_oficial,
            acao=aplicado.acao,
            condicao_satisfeita=aplicado.condicao_satisfeita,
            operador=aplicado.operador,
            limiar=aplicado.limiar,
            valor_observado=aplicado.valor_observado,
            origem_do_valor=aplicado.origem_do_valor,
            usa_modelo=aplicado.usa_modelo,
        )
        for aplicado in contexto.avaliacao_de_protocolos.aplicados
    ]


def _justificativas(
    contexto: ContextoDaExecucao,
) -> list[str]:
    """Justificativa objetiva por protocolo aplicado (condição satisfeita + valor)."""
    return [
        f"{protocolo.texto_oficial} -> condicao '{protocolo.condicao_satisfeita}' "
        f"satisfeita com valor observado {protocolo.valor_observado:g} "
        f"(origem: {protocolo.origem_do_valor})"
        for protocolo in _protocolos_do_contrato(contexto)
    ]


_PADRAO_DE_NUMERO = re.compile(r"\d+[.,]\d{2,}")


def _numeros_na_narrativa(narrativa: str | None) -> set[str]:
    """Números "com cara de probabilidade" presentes no texto do LLM."""
    if not narrativa:
        return set()
    return {item.replace(",", ".") for item in _PADRAO_DE_NUMERO.findall(narrativa)}


def verificar_narrativa(
    narrativa: str | None, probabilidade: float | None
) -> list[str]:
    """Compara o texto do Planejador com os valores autoritativos das tools.

    Se a narrativa citar um número com 2+ casas decimais diferente da probabilidade real, o
    pipeline **não** assume que está certo: devolve um aviso para o status virar
    ``CONCLUIDO_COM_RESSALVAS``. O valor oficial continua sendo o da tool.
    """
    avisos: list[str] = []
    citados = _numeros_na_narrativa(narrativa)
    if not citados:
        return avisos
    if probabilidade is None:
        avisos.append(
            "a narrativa cita valores numericos, mas a probabilidade nao foi calculada: "
            f"numeros citados={sorted(citados)}"
        )
        return avisos
    oficial = f"{probabilidade:.4f}"
    divergentes = sorted(numero for numero in citados if numero != oficial)
    if divergentes:
        avisos.append(
            "a narrativa do Planejador cita numero(s) divergente(s) da probabilidade "
            f"calculada pelo modelo ({divergentes} != {oficial}); o valor autoritativo e o "
            "da tool, nao o do texto"
        )
    return avisos


def verificar_handoffs_finais(nome_do_agente_final: str | None) -> list[str]:
    """Confere se o run terminou no agente esperado (o Planejador de Resposta)."""
    if nome_do_agente_final is None:
        return ["o run nao informou o agente final; o fluxo pode nao ter sido concluido"]
    if nome_do_agente_final != NOME_PLANEJADOR_DE_RESPOSTA:
        return [
            "o fluxo nao terminou no Planejador de Resposta "
            f"(agente final: {nome_do_agente_final}): algum handoff da cadeia nao ocorreu"
        ]
    return []


def _resposta_de_erro(
    identificador: str,
    erros: list[str],
    modo: risk_result.ModoDeExecucao,
    passos: list[risk_result.PassoDaExecucao] | None = None,
    contexto: ContextoDaExecucao | None = None,
) -> risk_result.RespostaDaExecucao:
    """Monta uma resposta ``ERRO``/``INCOMPLETO`` explícita (nunca um sucesso aparente)."""
    return risk_result.RespostaDaExecucao(
        id_execucao=identificador,
        status=risk_result.StatusDaExecucao.ERRO,
        modo=modo,
        erros=erros,
        passos=passos or [],
        handoffs_realizados=contexto.handoffs_realizados if contexto else [],
        agente_final=None,
    )


def _executar_run(
    agentes: AgentesDoPipeline,
    contexto: ContextoDaExecucao,
    modo: risk_result.ModoDeExecucao,
    provedor: Any | None,
    max_turnos: int,
    entrada: str,
    modelo: str | None = None,
) -> Any:
    """Dispara o ``Runner.run_sync`` do SDK com contexto, hooks e config explícitos.

    Nota importante (bug corrigido durante a validação): ``RunConfig.model_provider`` tem um
    *default factory*. Passar ``model_provider=None`` **sobrescreve** esse default e o SDK
    falha com ``AttributeError: 'NoneType' object has no attribute 'get_model'``. Por isso os
    campos são incluídos **só quando fazem sentido** (provider só no modo simulado; modelo
    só quando informado).
    """
    parametros: dict[str, Any] = {
        "tracing_disabled": True,  # nenhum dado sintetico vai para a plataforma da OpenAI
        "workflow_name": "CardioIA Fase 6 - pipeline multiagente",
        "trace_metadata": {"modo": modo.value},
    }
    if provedor is not None:
        parametros["model_provider"] = provedor
    if modelo:
        parametros["model"] = modelo

    configuracao = RunConfig(**parametros)
    return Runner.run_sync(
        agentes.orquestrador,
        entrada,
        context=contexto,
        max_turns=max_turnos,
        hooks=TrilhaDeExecucao(),
        run_config=configuracao,
    )


def _montar_resposta(
    contexto: ContextoDaExecucao,
    resultado: Any,
    identificador: str,
    modo: risk_result.ModoDeExecucao,
    duracao: float,
) -> risk_result.RespostaDaExecucao:
    """Monta o contrato a partir do contexto (valores autoritativos) + texto do Planejador."""
    narrativa, justificativas_do_llm, pontos_de_atencao = narrativa_como_texto(
        getattr(resultado, "final_output", None)
    )
    probabilidade = float(contexto.predicao["probabilidade_risco"]) if contexto.predicao else None

    avisos: list[str] = list(pontos_de_atencao)
    if modo is risk_result.ModoDeExecucao.SIMULADO:
        avisos.insert(0, AVISO_MODO_SIMULADO)
    avisos.extend(verificar_narrativa(narrativa, probabilidade))

    erros = list(contexto.erros)
    incompleto = verificar_handoffs_finais(getattr(getattr(resultado, "last_agent", None), "name", None))

    justificativas = _justificativas(contexto) or list(justificativas_do_llm)

    if incompleto:
        status = risk_result.StatusDaExecucao.INCOMPLETO
        erros.extend(incompleto)
    elif erros:
        status = (
            risk_result.StatusDaExecucao.CONCLUIDO_COM_RESSALVAS
            if contexto.tem_predicao
            else risk_result.StatusDaExecucao.INCOMPLETO
        )
    elif avisos and any("divergente" in aviso for aviso in avisos):
        status = risk_result.StatusDaExecucao.CONCLUIDO_COM_RESSALVAS
    else:
        status = risk_result.StatusDaExecucao.CONCLUIDO

    diagnostico = None
    if contexto.tem_predicao and contexto.predicao is not None:
        predicao = contexto.predicao
        diagnostico = risk_result.DiagnosticoDaPredicao(
            probabilidade_risco=predicao["probabilidade_risco"],
            pico_risco_previsto=int(predicao["pico_risco_previsto"]),
            classificacao_risco=predicao["classificacao_risco"],
            modelo_utilizado=predicao["modelo_utilizado"],
            classe_do_estimador=predicao["classe_do_estimador"],
            limiar_decisao=float(predicao["limiar_decisao"]),
            regra_de_decisao=predicao["regra_de_decisao"],
            features_utilizadas=list(predicao["features_utilizadas"]),
            origem_do_modelo=dict(predicao.get("origem_do_modelo", {})),
            avisos=list(predicao.get("avisos", [])),
        )

    if status is risk_result.StatusDaExecucao.CONCLUIDO and diagnostico is None:
        status = risk_result.StatusDaExecucao.INCOMPLETO
        erros.append("execucao sem predicao validada: nao pode ser considerada concluida")

    return risk_result.RespostaDaExecucao(
        id_execucao=identificador,
        status=status,
        modo=modo,
        diagnostico=diagnostico,
        indicadores=_indicadores_do_paciente(contexto.paciente),
        protocolos=_protocolos_do_contrato(contexto),
        justificativas=justificativas,
        narrativa=narrativa,
        avisos=avisos,
        erros=erros,
        passos=contexto.passos_do_contrato,
        handoffs_realizados=contexto.handoffs_realizados,
        agente_final=getattr(getattr(resultado, "last_agent", None), "name", None),
        versao_do_sdk=versao_do_sdk(),
        duracao_segundos=round(duracao, 4),
    )


def versao_do_sdk() -> str | None:
    """Versão instalada do OpenAI Agents SDK (``None`` se indisponível)."""
    try:
        import agents

        return str(getattr(agents, "__version__", "desconhecida"))
    except Exception:  # pragma: no cover - defensivo
        return None


# --------------------------------------------------------------------------- #
# Ponto de entrada do pipeline
# --------------------------------------------------------------------------- #


def executar_pipeline(
    paciente: Any,
    modo: risk_result.ModoDeExecucao | str = risk_result.ModoDeExecucao.SIMULADO,
    raiz: str | Path | None = None,
    pasta_modelos: str | Path | None = None,
    limiar: float = predict.LIMIAR_DE_DECISAO,
    modelo_simulado: simulacao.ModeloComRoteiro | None = None,
    max_turnos: int = MAX_TURNOS_PADRAO,
    entrada: str = ENTRADA_PADRAO,
    modelo: str | None = None,
) -> risk_result.RespostaDaExecucao:
    """Executa o pipeline completo e devolve o contrato validado.

    Ordem real de execução (adaptada às capacidades do SDK — ver docstring do módulo):

    ``Orquestrador`` -> handoff -> ``Analista de Risco`` -> tool do modelo ->
    handoff -> ``Especialista em Protocolos`` -> tool de protocolos -> handoff ->
    ``Planejador de Resposta`` (saída estruturada) -> montagem do contrato em Python.

    Args:
        paciente: dados do paciente (aceita a grafia do material da aula).
        modo: ``simulado`` (roteiro local, sem API) ou ``real`` (API da OpenAI).
        raiz: raiz do projeto (para localizar ``data/`` e ``models/``).
        pasta_modelos: pasta dos artefatos do modelo (útil em testes).
        limiar: limiar de decisão (operacional, repassado à tool de predição).
        modelo_simulado: modelo com roteiro a usar no modo simulado (permite testar falhas).
        max_turnos: limite de turnos do ``Runner`` (proteção contra laço).
        entrada: mensagem inicial enviada ao Orquestrador.
        modelo: nome do modelo de linguagem (opcional; útil no modo real, ex.: ``gpt-4o-mini``).
            Não tem efeito no modo simulado, que não chama a API.

    Returns:
        ``RespostaDaExecucao`` — com ``status=ERRO``/``INCOMPLETO`` quando algo falhar
        (nunca um sucesso aparente).
    """
    modo_efetivo = (
        modo if isinstance(modo, risk_result.ModoDeExecucao) else risk_result.ModoDeExecucao(modo)
    )
    identificador = str(uuid.uuid4())
    inicio = time.perf_counter()

    # 1) o modelo precisa existir ANTES de qualquer chamada de LLM (nunca treinar aqui).
    try:
        metadados = predict.carregar_metadados(pasta_modelos=pasta_modelos, raiz=raiz)
    except predict.ModeloNaoEncontradoError as erro:
        return _resposta_de_erro(
            identificador, [f"modelo indisponivel: {erro}"], modo_efetivo
        )

    # 2) validação da entrada do paciente (contrato do modelo, com avisos de faixa).
    try:
        valores, avisos_do_paciente = predict.validar_paciente(
            paciente,
            features=predict.features_do_modelo(metadados, None),
            faixas=metadados.get("faixas_esperadas"),
        )
    except predict.PacienteInvalidoError as erro:
        return _resposta_de_erro(
            identificador, [f"entrada do paciente invalida: {erro}"], modo_efetivo
        )

    # 3) modo real exige a chave (e ela nunca é gravada em código/log).
    if modo_efetivo is risk_result.ModoDeExecucao.REAL and not os.environ.get(VARIAVEL_DE_CHAVE):
        return _resposta_de_erro(identificador, [_ERRO_DE_CHAVE], modo_efetivo)

    contexto = ContextoDaExecucao(
        paciente=valores,
        raiz=Path(raiz) if raiz is not None else None,
        pasta_modelos=Path(pasta_modelos) if pasta_modelos is not None else None,
        limiar=limiar,
    )
    for aviso in avisos_do_paciente:
        contexto.registrar(
            risk_result.TipoDePasso.VALIDACAO, NOME_ORQUESTRADOR, f"aviso do paciente: {aviso}"
        )
    contexto.registrar(
        risk_result.TipoDePasso.VALIDACAO,
        NOME_ORQUESTRADOR,
        "entrada do paciente validada contra o contrato do modelo",
    )

    agentes = construir_agentes()
    if modo_efetivo is risk_result.ModoDeExecucao.SIMULADO:
        roteiro = modelo_simulado or simulacao.ModeloComRoteiro(roteiro=roteiro_padrao())
        provedor: Any | None = simulacao.ProvedorDeModelosSimulado(modelo=roteiro)
    else:
        roteiro = None
        provedor = None

    # 4) execução do run (handoffs e tools reais do SDK).
    try:
        resultado = _executar_run(
            agentes, contexto, modo_efetivo, provedor, max_turnos, entrada, modelo=modelo
        )
    except AgentsException as erro:
        return _resposta_de_erro(
            identificador,
            [f"falha do pipeline ({type(erro).__name__}): {erro}"],
            modo_efetivo,
            passos=contexto.passos_do_contrato,
            contexto=contexto,
        )
    except Exception as erro:  # pragma: no cover - falha inesperada
        return _resposta_de_erro(
            identificador,
            [f"falha inesperada ({type(erro).__name__}): {erro}"],
            modo_efetivo,
            passos=contexto.passos_do_contrato,
            contexto=contexto,
        )

    if roteiro is not None and roteiro.problemas:
        contexto.erros.extend(f"[roteiro simulado] {item}" for item in roteiro.problemas)

    # 5) montagem + validação do contrato (aqui os números são os das tools).
    return _montar_resposta(
        contexto,
        resultado,
        identificador,
        modo_efetivo,
        time.perf_counter() - inicio,
    )


__all__ = [
    "AVISO_MODO_SIMULADO",
    "ENTRADA_PADRAO",
    "INSTRUCOES_ORQUESTRADOR",
    "AgentesDoPipeline",
    "construir_agentes",
    "executar_pipeline",
    "roteiro_padrao",
    "verificar_handoffs_finais",
    "verificar_narrativa",
    "versao_do_sdk",
]