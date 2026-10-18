"""Interface de execução do CardioIA (Fase 6) — linha de comando.

Uso (a partir da raiz do repositório):

    python -m src.cardioia.main                          # paciente de data/novo_paciente.py
    python -m src.cardioia.main --json                   # saida JSON (contrato completo)
    python -m src.cardioia.main --modo real              # exige OPENAI_API_KEY no ambiente
    python -m src.cardioia.main --paciente caso.json     # paciente em arquivo JSON
    python -m src.cardioia.main --paciente '{"idade_media": 62, ...}'   # JSON inline

Comportamento em falha: **nunca** treina um modelo ausente, nunca inventa predição. Sai com
código 1 (e com o erro descrito) quando o pipeline não conclui.

Esta etapa não implementa API web nem interface gráfica (não são requisitos da atividade).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

from .agents import orchestrator
from .config import VARIAVEL_DE_MODELO
from .ml import dataset, predict
from .schemas import risk_result

CODIGO_SUCESSO = 0
CODIGO_FALHA = 1


def _configurar_saida_utf8() -> None:
    """Evita ``UnicodeEncodeError`` no console do Windows (cp1252) com o "→" dos protocolos."""
    for fluxo in (sys.stdout, sys.stderr):
        reconfigure = getattr(fluxo, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # pragma: no cover - console sem suporte
                pass


def _ler_paciente(origem: str | None, raiz: Path | None) -> dict[str, float]:
    """Carrega o paciente de um JSON inline/arquivo, ou o caso oficial da aula.

    Args:
        origem: ``None`` (caso oficial), JSON inline ou caminho de arquivo ``.json``.
        raiz: raiz do projeto (para localizar ``data/novo_paciente.py``).

    Raises:
        dataset.ArquivoOficialNaoEncontradoError: caso oficial ausente.
        ValueError: JSON inválido ou conteúdo não numérico.
    """
    if origem is None:
        return dataset.carregar_caso_do_paciente_da_aula(raiz=raiz)

    texto = origem
    candidato = Path(origem)
    if candidato.is_file():
        texto = candidato.read_text(encoding="utf-8")

    try:
        dados = json.loads(texto)
    except json.JSONDecodeError as erro:
        raise ValueError(
            f"'--paciente' deve ser JSON valido ou um arquivo .json existente: {erro}"
        ) from erro

    if not isinstance(dados, dict) or not dados:
        raise ValueError("'--paciente' deve ser um objeto JSON com os campos do paciente.")
    return dataset.normalizar_caso_de_paciente(dados)


def construir_analisador_de_argumentos() -> argparse.ArgumentParser:
    """Argumentos da linha de comando."""
    analisador = argparse.ArgumentParser(
        prog="python -m src.cardioia.main",
        description=(
            "Executa o pipeline multiagente do CardioIA (Fase 6): modelo de ML + base de "
            "protocolos + 4 agentes com handoffs. Dados SINTETICOS; sem validacao clinica."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exemplos:\n"
            "  python -m src.cardioia.main\n"
            "  python -m src.cardioia.main --json\n"
            "  python -m src.cardioia.main --paciente caso.json\n"
            "  python -m src.cardioia.main --modo real      (exige OPENAI_API_KEY)\n"
        ),
    )
    analisador.add_argument(
        "--modo",
        choices=[modo.value for modo in risk_result.ModoDeExecucao],
        default=risk_result.ModoDeExecucao.SIMULADO.value,
        help=(
            "simulado (padrao): roteiro local, sem chamar a API da OpenAI; "
            "real: usa a API da OpenAI (exige OPENAI_API_KEY)."
        ),
    )
    analisador.add_argument(
        "--paciente",
        default=None,
        help="JSON inline ou caminho de arquivo .json. Padrao: data/novo_paciente.py (material da aula).",
    )
    analisador.add_argument("--json", action="store_true", help="Imprime o contrato completo em JSON.")
    analisador.add_argument("--raiz", default=None, help="Raiz do projeto (padrao: descoberta automatica).")
    analisador.add_argument(
        "--pasta-modelos", default=None, help="Pasta dos artefatos do modelo (padrao: models/)."
    )
    analisador.add_argument(
        "--limiar",
        type=float,
        default=predict.LIMIAR_DE_DECISAO,
        help=f"Limiar de decisao (padrao {predict.LIMIAR_DE_DECISAO}; operacional, nao clinico).",
    )
    analisador.add_argument(
        "--max-turnos", type=int, default=10, help="Limite de turnos do Runner (padrao 10)."
    )
    analisador.add_argument("--quiet", action="store_true", help="Imprime apenas o status final.")
    analisador.add_argument(
        "--modelo",
        default=None,
        help=(
            "Modelo de linguagem do SDK (padrao: o do SDK). Util no modo real; "
            "alternativa: variavel de ambiente OPENAI_MODEL."
        ),
    )
    return analisador


def executar(argumentos: argparse.Namespace) -> int:
    """Executa o pipeline conforme os argumentos e devolve o código de saída."""
    raiz = Path(argumentos.raiz) if argumentos.raiz else dataset.localizar_raiz_do_projeto()
    pasta_modelos = Path(argumentos.pasta_modelos) if argumentos.pasta_modelos else None

    # Pré-checagem explícita: o modelo precisa existir (nunca treinamos silenciosamente).
    try:
        metadados = predict.carregar_metadados(pasta_modelos=pasta_modelos, raiz=raiz)
    except predict.ModeloNaoEncontradoError as erro:
        print(f"[erro] modelo indisponivel. {erro}", file=sys.stderr)
        print(
            "[dica] treine o modelo com: python -m src.cardioia.ml.train",
            file=sys.stderr,
        )
        return CODIGO_FALHA

    try:
        paciente = _ler_paciente(argumentos.paciente, raiz)
    except (ValueError, dataset.ErroDeDados) as erro:
        print(f"[erro] nao foi possivel ler os dados do paciente: {erro}", file=sys.stderr)
        return CODIGO_FALHA

    if not argumentos.quiet and not argumentos.json:
        origem_do_paciente = (
            "data/novo_paciente.py (material da aula)"
            if argumentos.paciente is None
            else "informado em --paciente"
        )
        print(f"[info] paciente: {origem_do_paciente}")
        print(
            f"[info] modelo: {metadados.get('nome_do_modelo')} | "
            f"limiar: {argumentos.limiar} | modo: {argumentos.modo}"
        )
        print()

    resposta = orchestrator.executar_pipeline(
        paciente,
        modo=argumentos.modo,
        raiz=raiz,
        pasta_modelos=pasta_modelos,
        limiar=argumentos.limiar,
        max_turnos=argumentos.max_turnos,
        modelo=argumentos.modelo or os.environ.get(VARIAVEL_DE_MODELO) or None,
    )

    if argumentos.json:
        print(json.dumps(resposta.como_dicionario(), ensure_ascii=False, indent=2))
    elif argumentos.quiet:
        print(f"status={resposta.status.value} id={resposta.id_execucao}")
    else:
        print(resposta.resumo_textual())

    if resposta.erros:
        print("", file=sys.stderr)
        for erro in resposta.erros:
            print(f"[erro] {erro}", file=sys.stderr)

    return CODIGO_SUCESSO if resposta.concluida else CODIGO_FALHA


def main(argv: Sequence[str] | None = None) -> int:
    """Ponto de entrada da CLI."""
    _configurar_saida_utf8()
    argumentos = construir_analisador_de_argumentos().parse_args(argv)
    try:
        return executar(argumentos)
    except KeyboardInterrupt:  # pragma: no cover - interacao do usuario
        print("\n[erro] execucao interrompida pelo usuario.", file=sys.stderr)
        return CODIGO_FALHA
    except Exception as erro:  # pragma: no cover - falha inesperada
        print(f"[erro] falha inesperada ({type(erro).__name__}): {erro}", file=sys.stderr)
        return CODIGO_FALHA


if __name__ == "__main__":  # pragma: no cover - execucao via linha de comando
    raise SystemExit(main())


__all__ = ["construir_analisador_de_argumentos", "executar", "main"]