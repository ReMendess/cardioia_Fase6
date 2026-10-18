"""Testes da interface de linha de comando (``src/cardioia/main.py``).

Os testes chamam o **processo real** (``python -m src.cardioia.main``) a partir da raiz do
repositório: assim cobrem exatamente o que o usuário executa, incluindo o **código de saída**.
Nenhum deles usa a API da OpenAI (modo simulado) nem precisa de ``OPENAI_API_KEY``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

pytest.importorskip("agents", reason="OpenAI Agents SDK nao instalado")

from cardioia.ml import dataset, predict  # noqa: E402
from cardioia.schemas import risk_result  # noqa: E402

TEMPO_LIMITE = 180


def _ambiente_sem_chave() -> dict[str, str]:
    """Ambiente sem ``OPENAI_API_KEY``: prova que o modo simulado não depende de créditos."""
    ambiente = os.environ.copy()
    ambiente.pop("OPENAI_API_KEY", None)
    ambiente.pop("OPENAI_MODEL", None)
    ambiente["PYTHONIOENCODING"] = "utf-8"
    return ambiente


def _executar(raiz, *argumentos: str) -> subprocess.CompletedProcess:
    """Executa a CLI em um **processo novo**, a partir da raiz do repositório.

    Usado apenas nos testes que precisam do comando real e do **código de saída**; o restante
    roda in-process (``main(argv)`` + ``capsys``) para não pagar a inicialização do
    interpretador em cada caso.
    """
    return subprocess.run(
        [sys.executable, "-m", "src.cardioia.main", *argumentos],
        cwd=str(raiz),
        env=_ambiente_sem_chave(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=TEMPO_LIMITE,
        check=False,
    )


def _executar_inprocess(argumentos: list[str], capsys) -> tuple[int, str, str]:
    """Chama ``main(argv)`` no mesmo processo e devolve ``(codigo, stdout, stderr)``."""
    from cardioia.main import main

    codigo = main(argumentos)
    capturado = capsys.readouterr()
    return codigo, capturado.out, capturado.err


@pytest.fixture(scope="module")
def argumentos_do_modelo(modelo_temporario, raiz_do_projeto) -> list[str]:
    """Aponta a CLI para o modelo da fixture (diretório temporário)."""
    return ["--raiz", str(raiz_do_projeto), "--pasta-modelos", str(modelo_temporario["pasta"])]


# --------------------------------------------------------------------------- #
# Caminho feliz
# --------------------------------------------------------------------------- #


def test_execucao_simulada_conclui_e_sai_com_zero(capsys, argumentos_do_modelo):
    codigo, saida, erros = _executar_inprocess(["--quiet", *argumentos_do_modelo], capsys)
    assert codigo == 0, erros
    assert "status=CONCLUIDO" in saida


def test_cli_real_em_processo_novo_conclui(raiz_do_projeto, argumentos_do_modelo):
    """Comando real (``python -m src.cardioia.main``) com o código de saída do sistema."""
    resultado = _executar(raiz_do_projeto, "--quiet", *argumentos_do_modelo)
    assert resultado.returncode == 0, resultado.stderr
    assert "status=CONCLUIDO" in resultado.stdout


def test_modo_simulado_nao_exige_chave_de_api(capsys, monkeypatch, argumentos_do_modelo):
    """Sem ``OPENAI_API_KEY`` no ambiente, o modo simulado (padrão) ainda conclui."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    codigo, saida, erros = _executar_inprocess(["--quiet", *argumentos_do_modelo], capsys)
    assert codigo == 0
    assert "OPENAI_API_KEY" not in saida
    assert "OPENAI_API_KEY" not in erros


def test_saida_json_e_valida_e_respeita_o_contrato(capsys, argumentos_do_modelo):
    codigo, saida, erros = _executar_inprocess(["--json", *argumentos_do_modelo], capsys)
    assert codigo == 0, erros

    dados = json.loads(saida)  # 1) é JSON válido
    # 2) revalidação pelo contrato Pydantic (o JSON da CLI é o mesmo contrato dos agentes)
    resposta = risk_result.RespostaDaExecucao(**dados)
    assert resposta.status is risk_result.StatusDaExecucao.CONCLUIDO
    assert resposta.modo is risk_result.ModoDeExecucao.SIMULADO
    assert resposta.diagnostico is not None
    assert 0.0 <= (resposta.diagnostico.probabilidade_risco or 0.0) <= 1.0
    assert resposta.agente_final == "Planejador de Resposta"
    assert len(resposta.handoffs_realizados) == 3
    assert resposta.limitacoes and resposta.avisos


def test_json_e_textual_coincidem_no_status(capsys, argumentos_do_modelo):
    _, textual, _ = _executar_inprocess(["--quiet", *argumentos_do_modelo], capsys)
    _, estruturado, _ = _executar_inprocess(["--json", *argumentos_do_modelo], capsys)
    dados = json.loads(estruturado)
    assert f"status={dados['status']}" in textual


# --------------------------------------------------------------------------- #
# Apresentação no modo simulado (LIMITACOES/AVISOS/ERROS fora do texto)
# --------------------------------------------------------------------------- #


def test_saida_simulada_omite_limitacoes_avisos_e_erros(capsys, argumentos_do_modelo):
    """Modo simulado concluído: o texto sai enxuto e o ``stderr`` fica limpo."""
    codigo, saida, erros = _executar_inprocess([*argumentos_do_modelo], capsys)
    assert codigo == 0, erros
    assert "PREDICAO" in saida
    assert "PROTOCOLOS APLICAVEIS (4)" in saida
    for secao in risk_result.SECOES_OCULTAS_NO_MODO_SIMULADO:
        assert secao not in saida, f"secao impressa no modo simulado: {secao}"
    assert erros.strip() == ""


def test_json_continua_completo_mesmo_com_o_texto_enxuto(capsys, argumentos_do_modelo):
    """A supressão é só de apresentação: o contrato (JSON) segue íntegro e auditável."""
    codigo, saida, erros = _executar_inprocess(["--json", *argumentos_do_modelo], capsys)
    assert codigo == 0, erros
    dados = json.loads(saida)
    assert len(dados["limitacoes"]) == len(risk_result.LIMITACOES_PADRAO)
    assert dados["avisos"]
    assert dados["passos"] and dados["handoffs_realizados"]


def test_falha_no_modo_simulado_mostra_os_erros(capsys, argumentos_do_modelo):
    """Exceção da regra: execução que NÃO conclui continua explicando a falha (saída 1)."""
    codigo, saida, erros = _executar_inprocess(
        ["--max-turnos", "1", *argumentos_do_modelo], capsys
    )
    assert codigo == 1
    assert "ERROS" in saida
    assert "MaxTurnsExceeded" in saida  # o motivo aparece no texto...
    assert "MaxTurnsExceeded" in erros  # ...e também no stderr


def test_paciente_por_arquivo_json(capsys, argumentos_do_modelo, paciente_sem_risco, tmp_path):
    caminho = tmp_path / "paciente.json"
    caminho.write_text(json.dumps(paciente_sem_risco), encoding="utf-8")
    codigo, saida, erros = _executar_inprocess(
        ["--json", "--paciente", str(caminho), *argumentos_do_modelo], capsys
    )
    assert codigo == 0, erros
    dados = json.loads(saida)
    assert dados["status"] == "CONCLUIDO"
    assert dados["protocolos"] == []  # paciente sem risco não aciona protocolo
    assert dados["diagnostico"]["pico_risco_previsto"] == 0


def test_json_do_paciente_invalido_gera_erro_de_leitura(capsys, argumentos_do_modelo):
    codigo, _, erros = _executar_inprocess(
        ["--quiet", "--paciente", "{isso nao e json}", *argumentos_do_modelo], capsys
    )
    assert codigo == 1
    assert "nao foi possivel ler os dados do paciente" in erros
    assert "Traceback" not in erros


# --------------------------------------------------------------------------- #
# Falhas: código de saída e ausência de falso sucesso
# --------------------------------------------------------------------------- #


def test_modelo_ausente_sai_com_erro_e_nao_finge_sucesso(capsys, raiz_do_projeto, tmp_path):
    pasta_vazia = tmp_path / "sem_modelo"
    pasta_vazia.mkdir()
    codigo, saida, erros = _executar_inprocess(
        ["--quiet", "--raiz", str(raiz_do_projeto), "--pasta-modelos", str(pasta_vazia)], capsys
    )
    assert codigo == 1
    assert "modelo indisponivel" in erros
    assert "python -m src.cardioia.ml.train" in erros  # instrução de correção
    assert "status=CONCLUIDO" not in saida
    assert list(pasta_vazia.iterdir()) == []  # nunca treina silenciosamente


def test_modo_real_sem_chave_sai_com_erro(capsys, monkeypatch, argumentos_do_modelo):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    codigo, saida, erros = _executar_inprocess(
        ["--quiet", "--modo", "real", *argumentos_do_modelo], capsys
    )
    assert codigo == 1
    assert "OPENAI_API_KEY" in (erros + saida)
    assert "status=ERRO" in saida


def test_modo_invalido_e_rejeitado_pelo_argparse(capsys):
    from cardioia.main import main

    with pytest.raises(SystemExit) as saida:
        main(["--modo", "inventado"])
    assert saida.value.code == 2  # código padrão do argparse para erro de uso
    assert "invalid choice" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# Contrato da CLI (guarda de regressão sobre as flags realmente implementadas)
# --------------------------------------------------------------------------- #


def _texto_do_help(capsys) -> str:
    from cardioia.main import main

    with pytest.raises(SystemExit) as saida:
        main(["--help"])
    assert saida.value.code == 0
    return capsys.readouterr().out


def test_help_lista_as_flags_reais(capsys):
    texto = _texto_do_help(capsys)
    for flag in (
        "--modo",
        "--paciente",
        "--json",
        "--raiz",
        "--pasta-modelos",
        "--limiar",
        "--max-turnos",
        "--quiet",
        "--modelo",
    ):
        assert flag in texto, f"flag ausente no --help: {flag}"


def test_help_documenta_a_limitacao_dos_dados_sinteticos(capsys):
    texto = _texto_do_help(capsys)
    assert "SINTETICOS" in texto
    assert "sem validacao clinica" in texto


# --------------------------------------------------------------------------- #
# Independência do diretório de execução (caminhos relativos)
# --------------------------------------------------------------------------- #


def test_cli_nao_depende_do_diretorio_de_execucao(tmp_path, modelo_temporario, raiz_do_projeto):
    """Executada de outro diretório e com ``--raiz``, a CLI continua funcionando."""
    resultado = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.cardioia.main",
            "--quiet",
            "--raiz",
            str(raiz_do_projeto),
            "--pasta-modelos",
            str(modelo_temporario["pasta"]),
        ],
        cwd=str(tmp_path),
        env=_ambiente_sem_chave(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=TEMPO_LIMITE,
        check=False,
    )
    # Sem PYTHONPATH, rodar "src.cardioia.main" fora da raiz não é possível; o que se valida
    # aqui é que a CLI NÃO grava nada por engano no diretório corrente.
    assert not any(tmp_path.iterdir()), "a CLI criou arquivos no diretorio de execucao"
    if resultado.returncode == 0:
        assert "status=CONCLUIDO" in resultado.stdout


def test_execucao_da_raiz_nao_polui_o_repositorio(raiz_do_projeto, argumentos_do_modelo):
    """Rodar a CLI na raiz não cria artefatos indevidos (só usa ``models/``, que já existe)."""
    antes = {caminho.name for caminho in raiz_do_projeto.iterdir()}
    assert _executar(raiz_do_projeto, "--quiet", *argumentos_do_modelo).returncode == 0
    depois = {caminho.name for caminho in raiz_do_projeto.iterdir()}
    assert depois == antes, f"a CLI criou itens novos na raiz: {depois - antes}"
    assert dataset.localizar_raiz_do_projeto(raiz_do_projeto) == raiz_do_projeto
    assert predict.NOME_ARQUIVO_MODELO in {c.name for c in (raiz_do_projeto / "models").iterdir()}