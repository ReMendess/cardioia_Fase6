"""Teste de integração **real** com a API da OpenAI (opcional).

Este arquivo é o único que chama o serviço externo. Ele é **pulado** quando a variável de
ambiente ``OPENAI_API_KEY`` não está definida — o que é o caso deste ambiente. Portanto:

- a suíte obrigatória (``test_dataset``, ``test_prediction``, ``test_protocols``,
  ``test_schemas``, ``test_agents``, ``test_orchestrator``) roda **sem** rede e sem chave;
- a integração com o serviço externo **não foi validada** por mim. Quando houver chave, este
  teste executa o pipeline completo com o modelo real (``--modo real``).

Como rodar com a chave:

    $env:OPENAI_API_KEY = "sk-..."
    python -m pytest tests/test_integracao_real.py -v -m integracao
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("agents", reason="OpenAI Agents SDK nao instalado")

from cardioia.agents import orchestrator  # noqa: E402
from cardioia.ml import predict  # noqa: E402
from cardioia.schemas import risk_result  # noqa: E402

TEM_CHAVE = bool(os.environ.get("OPENAI_API_KEY"))
MODELO_PADRAO = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

pytestmark = [
    pytest.mark.integracao,
    pytest.mark.skipif(
        not TEM_CHAVE,
        reason="OPENAI_API_KEY nao definida: integracao com a API real nao executada.",
    ),
]


def _executar_modo_real(paciente, modelo_temporario, raiz_do_projeto):
    """Executa o pipeline no modo real; se a conta não puder pagar, **pula** com o motivo.

    Um erro de conta (429/`insufficient_quota`) ou de credencial **não** deve ser mascarado
    como sucesso: transformamos em ``skip`` com a mensagem original da API, para que a
    limitação fique visível sem poluir a suíte com uma falha de ambiente.
    """
    resposta = orchestrator.executar_pipeline(
        paciente,
        modo=risk_result.ModoDeExecucao.REAL,
        raiz=raiz_do_projeto,
        pasta_modelos=modelo_temporario["pasta"],
    )
    texto_dos_erros = " ".join(resposta.erros)
    for marcador, motivo in (
        ("insufficient_quota", "conta sem creditos (HTTP 429 / insufficient_quota)"),
        ("credit_balance_exhausted", "conta sem creditos (credit_balance_exhausted)"),
        ("RateLimitError", "limite de requisicoes da API (HTTP 429)"),
        ("AuthenticationError", "credencial recusada pela API (HTTP 401)"),
        ("invalid_api_key", "chave de API invalida"),
    ):
        if marcador in texto_dos_erros:
            pytest.skip(f"Integracao real NAO validada: {motivo}. Erro da API: {texto_dos_erros}")
    return resposta


def test_pipeline_com_a_api_real(modelo_temporario, paciente_da_aula, raiz_do_projeto):
    """Executa o fluxo completo com o LLM real e valida o contrato de saída.

    Observações:
        - o modelo de ML usado é o artefato da fixture (diretório temporário), então o teste
          não depende do ``.pkl`` versionado;
        - ``tracing_disabled=True`` é mantido (nenhum dado sintético vai para a plataforma).
    """
    resposta = _executar_modo_real(paciente_da_aula, modelo_temporario, raiz_do_projeto)

    assert resposta.modo is risk_result.ModoDeExecucao.REAL
    assert resposta.concluida is True, f"status={resposta.status.value} erros={resposta.erros}"
    assert resposta.diagnostico is not None

    esperado = predict.prever_pico_risco(
        paciente_da_aula, pasta_modelos=modelo_temporario["pasta"], raiz=raiz_do_projeto
    )
    # A probabilidade continua vindo da tool em Python, não do LLM.
    assert resposta.diagnostico.probabilidade_risco == esperado["probabilidade_risco"]
    assert resposta.handoffs_realizados == [
        "Orquestrador -> Analista de Risco",
        "Analista de Risco -> Especialista em Protocolos",
        "Especialista em Protocolos -> Planejador de Resposta",
    ]
    assert resposta.agente_final == "Planejador de Resposta"


def test_llm_nao_altera_a_probabilidade(modelo_temporario, paciente_da_aula, raiz_do_projeto):
    """Mesmo com o LLM real na rota, os números vêm do modelo de ML."""
    resposta = _executar_modo_real(paciente_da_aula, modelo_temporario, raiz_do_projeto)
    if not resposta.concluida or resposta.diagnostico is None:
        pytest.skip(f"pipeline nao concluiu na execucao real: {resposta.status.value}")

    esperado = predict.prever_pico_risco(
        paciente_da_aula, pasta_modelos=modelo_temporario["pasta"], raiz=raiz_do_projeto
    )
    assert resposta.diagnostico.probabilidade_risco == esperado["probabilidade_risco"]
    assert not any("divergente" in aviso for aviso in resposta.avisos)


def test_modo_real_sem_creditos_reporta_erro_e_nao_sucesso(
    modelo_temporario, paciente_da_aula, raiz_do_projeto
):
    """Sem crédito/credencial, o pipeline deve falhar de forma explícita (nunca fingir sucesso)."""
    resposta = orchestrator.executar_pipeline(
        paciente_da_aula,
        modo=risk_result.ModoDeExecucao.REAL,
        raiz=raiz_do_projeto,
        pasta_modelos=modelo_temporario["pasta"],
    )
    if resposta.concluida:
        pytest.skip("conta com creditos: o cenario de falha de API nao pode ser exercitado")
    assert resposta.diagnostico is None
    assert resposta.erros  # o motivo fica registrado, nunca escondido