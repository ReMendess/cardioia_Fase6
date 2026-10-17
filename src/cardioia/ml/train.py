"""Treino, comparação e persistência do modelo de ``pico_risco`` (CardioIA — Fase 6).

O treinamento é executável pela linha de comando a partir da raiz do repositório:

    python -m src.cardioia.ml.train
    python -m src.cardioia.ml.train --pasta-modelos models --semente 42
    python -m src.cardioia.ml.train --dry-run        # mede e NAO grava artefatos

Decisões preservadas do notebook (não alterar sem justificativa registrada):

- semente 42 e divisão estratificada 80/20, iguais para todos os modelos;
- **mesmos** conjuntos de treino e teste para os modelos comparados;
- o conjunto de teste é usado **uma única vez**, apenas para medir e aplicar o critério
  pré-declarado (nenhum ajuste de hiperparâmetro é feito olhando o teste);
- critério de seleção: maior recall da classe ``pico_risco=1``; empate por acurácia;
  empate final por interpretabilidade.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from . import dataset, evaluate
from .dataset import COLUNAS_PREDITORAS, SEMENTE, TAMANHO_TESTE, VARIAVEL_ALVO
from .predict import LIMIAR_DE_DECISAO

NOME_ARQUIVO_MODELO = "modelo_pico_risco.pkl"
NOME_ARQUIVO_METADADOS = "modelo_pico_risco_metadata.json"
PASTA_MODELOS_PADRAO = "models"
N_ESTIMATORS_PADRAO = 300

NOME_MODELO_INTERPRETAVEL = "Regressao Logistica"
NOME_MODELO_NAO_LINEAR = "Random Forest"

CHAVES_DE_PARAMETROS = (
    "max_iter",
    "random_state",
    "n_estimators",
    "n_jobs",
    "C",
    "penalty",
    "solver",
    "max_depth",
)


class ErroDeTreino(Exception):
    """Falha no fluxo de treino (dados, modelo desconhecido ou persistência)."""


def _parametros_uteis(estimador: Any) -> dict[str, Any]:
    """Filtra os hiperparâmetros relevantes para o metadata (não despeja o default todo)."""
    parametros = estimador.get_params()
    uteis: dict[str, Any] = {}
    for chave in CHAVES_DE_PARAMETROS:
        if chave in parametros:
            valor = parametros[chave]
            uteis[chave] = valor if isinstance(valor, (int, float, str, bool, type(None))) else str(valor)
    return uteis


def descrever_parametros(modelo: Any) -> dict[str, Any]:
    """Descreve os hiperparâmetros de forma serializável em JSON."""
    if isinstance(modelo, Pipeline):
        return {
            "tipo": "Pipeline",
            "passos": [
                {"passo": nome, "classe": type(passo).__name__, "parametros": _parametros_uteis(passo)}
                for nome, passo in modelo.steps
            ],
        }
    return {"tipo": type(modelo).__name__, "parametros": _parametros_uteis(modelo)}


def construir_modelos(
    semente: int = SEMENTE, n_estimators: int = N_ESTIMATORS_PADRAO
) -> dict[str, Any]:
    """Cria os mesmos classificadores do notebook, com os mesmos hiperparâmetros."""
    if n_estimators <= 0:
        raise ErroDeTreino(f"'n_estimators' deve ser positivo; recebido {n_estimators!r}.")
    return {
        NOME_MODELO_INTERPRETAVEL: make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=semente),
        ),
        NOME_MODELO_NAO_LINEAR: RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=semente,
            n_jobs=-1,
        ),
    }


def montar_metadados(
    resultado: Mapping[str, Any],
    origem_da_base: str | None = None,
    pasta_modelos: str | Path | None = None,
) -> dict[str, Any]:
    """Monta o JSON de metadados do modelo escolhido.

    Mantém as chaves já usadas na Etapa 1 (``nome_do_modelo``, ``features``,
    ``metricas_no_teste``, ``sha256_da_base_usada``, ``versoes``, ``aviso``) e acrescenta
    rastreabilidade extra: parâmetros do estimador, limiar de decisão, faixas observadas
    e a lista de colunas derivadas do alvo que ficaram fora das features.
    """
    df = resultado["base"]
    escolhido_nome = resultado["nome_modelo_escolhido"]
    escolhido = resultado["modelo_escolhido"]

    return {
        "nome_do_modelo": escolhido_nome,
        "classe_do_estimador": type(escolhido).__name__,
        "variavel_alvo": VARIAVEL_ALVO,
        "features": list(COLUNAS_PREDITORAS),
        "colunas_derivadas_do_alvo_excluidas": list(dataset.COLUNAS_AUXILIARES_IGNORADAS),
        "criterio_de_escolha": evaluate.CRITERIO_DE_SELECAO,
        "parametros_do_estimador": descrever_parametros(escolhido),
        "limiar_de_decisao": LIMIAR_DE_DECISAO,
        "limiar_de_decisao_observacao": (
            "Limiar operacional do prototipo (0.5). NAO e um limite clinico validado."
        ),
        "semente": int(resultado["semente"]),
        "tamanho_do_teste": float(resultado["tamanho_teste"]),
        "split_estratificado": True,
        "amostras_total": int(len(df)),
        "amostras_treino": int(len(resultado["X_treino"])),
        "amostras_teste": int(len(resultado["X_teste"])),
        "metricas_no_teste": {
            nome: {
                "acuracia": avaliacao["acuracia"],
                "matriz_confusao": avaliacao["matriz_confusao"],
                "recall_classe_1": avaliacao["recall_classe_1"],
                "precisao_classe_1": avaliacao["precisao_classe_1"],
                "f1_classe_1": avaliacao["f1_classe_1"],
                "falsos_negativos": avaliacao["falsos_negativos"],
                "falsos_positivos": avaliacao["falsos_positivos"],
                "macro_f1": avaliacao["macro"]["f1"],
            }
            for nome, avaliacao in resultado["avaliacoes"].items()
        },
        "acuracia_da_regra_deterministica_no_teste": resultado["acuracia_da_regra"],
        "origem_da_base": origem_da_base or "arquivo oficial data/base_dados.py",
        "sha256_da_base_usada": dataset.sha256_da_base(df),
        "faixas_esperadas": dataset.faixas_observadas(df),
        "versoes": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "gerado_por": "src/cardioia/ml/train.py",
        "pasta_modelos": str(pasta_modelos) if pasta_modelos else PASTA_MODELOS_PADRAO,
        "aviso": dataset.AVISO_DADOS_SINTETICOS,
    }


def treinar_e_avaliar(
    df: pd.DataFrame | None = None,
    semente: int = SEMENTE,
    tamanho_teste: float = TAMANHO_TESTE,
    n_estimators: int = N_ESTIMATORS_PADRAO,
    raiz: str | Path | None = None,
    origem_da_base: str | None = None,
) -> dict[str, Any]:
    """Treina os classificadores, mede no teste e aplica o critério pré-declarado.

    Nada é gravado em disco aqui: use :func:`salvar_artefatos` (ou :func:`treinar`).

    Args:
        df: base já carregada. Se ``None``, carrega o material oficial.
        semente: semente do split e dos modelos.
        tamanho_teste: fração do conjunto de teste.
        n_estimators: número de árvores do Random Forest.
        raiz: raiz do projeto (para localizar ``data/base_dados.py``).
        origem_da_base: texto de rastreabilidade da base (vai para o metadata).

    Returns:
        Dicionário com base, partições, modelos treinados, avaliações, comparativo,
        modelo escolhido e metadados.
    """
    if df is None:
        df, informacoes = dataset.carregar_base_oficial(raiz=raiz, retornar_origem=True)
        origem_da_base = origem_da_base or str(informacoes["origem"])
    else:
        dataset.validar_base(df)

    X, y = dataset.separar_features_e_alvo(df)
    X_treino, X_teste, y_treino, y_teste = dataset.divisao_treino_teste(
        X, y, semente=semente, tamanho_teste=tamanho_teste
    )

    modelos = construir_modelos(semente=semente, n_estimators=n_estimators)
    avaliacoes: dict[str, dict[str, Any]] = {}
    for nome, modelo in modelos.items():
        modelo.fit(X_treino, y_treino)  # mesmo conjunto de treino para todos
        avaliacoes[nome] = evaluate.avaliar_modelo(modelo, X_teste, y_teste, nome_do_modelo=nome)

    comparativo = evaluate.comparar_modelos(avaliacoes)
    if not comparativo:
        raise ErroDeTreino("Nenhum modelo foi treinado/avaliado.")
    escolhido_nome = str(comparativo[0]["modelo"])

    # Oráculo determinístico: teto de referência (a regra define o próprio rótulo).
    regra_no_teste = dataset.rotulo_pela_regra(df).loc[X_teste.index]
    acuracia_da_regra = evaluate.acuracia_da_regra_deterministica(y_teste, regra_no_teste)

    resultado: dict[str, Any] = {
        "base": df,
        "X": X,
        "y": y,
        "X_treino": X_treino,
        "X_teste": X_teste,
        "y_treino": y_treino,
        "y_teste": y_teste,
        "modelos": modelos,
        "avaliacoes": avaliacoes,
        "comparativo": comparativo,
        "nome_modelo_escolhido": escolhido_nome,
        "modelo_escolhido": modelos[escolhido_nome],
        "regra_deterministica_no_teste": regra_no_teste,
        "acuracia_da_regra": acuracia_da_regra,
        "semente": semente,
        "tamanho_teste": tamanho_teste,
        "n_estimators": n_estimators,
        "origem_da_base": origem_da_base or "arquivo oficial data/base_dados.py",
    }
    resultado["metadados"] = montar_metadados(
        resultado, origem_da_base=resultado["origem_da_base"]
    )
    return resultado


def resolver_pasta_modelos(pasta_modelos: str | Path | None = None, raiz: str | Path | None = None) -> Path:
    """Resolve (e cria) a pasta de artefatos. Padrão: ``models/`` na raiz do projeto."""
    if pasta_modelos is not None:
        destino = Path(pasta_modelos)
    else:
        raiz_resolvida = Path(raiz) if raiz is not None else dataset.localizar_raiz_do_projeto()
        destino = (raiz_resolvida or Path.cwd()) / PASTA_MODELOS_PADRAO
    destino.mkdir(parents=True, exist_ok=True)
    return destino


def salvar_artefatos(
    resultado: Mapping[str, Any],
    pasta_modelos: str | Path | None = None,
    raiz: str | Path | None = None,
) -> dict[str, Any]:
    """Grava o modelo escolhido (``joblib``) e o JSON de metadados.

    Returns:
        Dicionário com ``pasta_modelos``, ``caminho_modelo``, ``caminho_metadados``,
        ``metadados`` e os tamanhos gravados.
    """
    if "modelo_escolhido" not in resultado or "metadados" not in resultado:
        raise ErroDeTreino(
            "Resultado de treino incompleto para salvamento: rode 'treinar_e_avaliar' antes."
        )
    destino = resolver_pasta_modelos(pasta_modelos, raiz=raiz)
    caminho_modelo = destino / NOME_ARQUIVO_MODELO
    caminho_metadados = destino / NOME_ARQUIVO_METADADOS

    metadados = dict(resultado["metadados"])
    metadados["pasta_modelos"] = str(destino)

    joblib.dump(resultado["modelo_escolhido"], caminho_modelo)
    caminho_metadados.write_text(
        json.dumps(metadados, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return {
        "pasta_modelos": destino,
        "caminho_modelo": caminho_modelo,
        "caminho_metadados": caminho_metadados,
        "tamanho_modelo_bytes": caminho_modelo.stat().st_size,
        "tamanho_metadados_bytes": caminho_metadados.stat().st_size,
        "metadados": metadados,
    }


def treinar(
    salvar: bool = True,
    pasta_modelos: str | Path | None = None,
    raiz: str | Path | None = None,
    semente: int = SEMENTE,
    tamanho_teste: float = TAMANHO_TESTE,
    n_estimators: int = N_ESTIMATORS_PADRAO,
    df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Fluxo completo: carrega a base, treina, mede, escolhe e (opcionalmente) grava.

    Args:
        salvar: se ``False``, apenas mede (``--dry-run`` da CLI).
        pasta_modelos: destino dos artefatos. Padrão: ``models/`` na raiz do projeto.
        raiz: raiz do projeto (descoberta automaticamente quando ``None``).
        semente: semente do split e dos modelos.
        tamanho_teste: fração do teste (estratificado).
        n_estimators: número de árvores do Random Forest.
        df: base já carregada (usado nos testes e no notebook).

    Returns:
        O dicionário de :func:`treinar_e_avaliar`; com ``salvar=True``, acrescido de
        ``artefatos``.
    """
    resultado = treinar_e_avaliar(
        df=df,
        semente=semente,
        tamanho_teste=tamanho_teste,
        n_estimators=n_estimators,
        raiz=raiz,
    )
    if salvar:
        resultado["artefatos"] = salvar_artefatos(resultado, pasta_modelos=pasta_modelos, raiz=raiz)
    return resultado


def formatar_resumo(resultado: Mapping[str, Any]) -> str:
    """Resumo textual do treino (usado pela CLI e pelo notebook)."""
    linhas = [
        "=" * 72,
        "Treino do modelo de pico_risco (CardioIA - Fase 6)",
        "=" * 72,
        f"  origem da base             : {resultado['origem_da_base']}",
        f"  semente                    : {resultado['semente']}",
        f"  amostras (treino/teste)    : {len(resultado['X_treino'])} / {len(resultado['X_teste'])}",
        f"  positivos no teste         : {int(resultado['y_teste'].sum())}",
        f"  criterio de escolha        : {evaluate.CRITERIO_DE_SELECAO}",
        "",
        "  Comparativo no MESMO conjunto de teste (ordenado pelo criterio):",
    ]
    for linha in resultado["comparativo"]:
        linhas.append(
            "    {modelo:<20s} acuracia={acuracia} recall(1)={recall} precisao(1)={precisao} "
            "FN={fn} FP={fp}".format(
                modelo=linha["modelo"],
                acuracia=linha["acuracia"],
                recall=linha["recall_classe_1"],
                precisao=linha["precisao_classe_1"],
                fn=linha["falsos_negativos"],
                fp=linha["falsos_positivos"],
            )
        )
    linhas += [
        f"    {'Regra deterministica':<20s} acuracia={resultado['acuracia_da_regra']} (teto, por construcao)",
        "",
        f"  Modelo escolhido           : {resultado['nome_modelo_escolhido']}",
    ]
    artefatos = resultado.get("artefatos")
    if artefatos:
        linhas += [
            f"  Modelo salvo em            : {artefatos['caminho_modelo']} "
            f"({artefatos['tamanho_modelo_bytes']} bytes)",
            f"  Metadados salvos em        : {artefatos['caminho_metadados']} "
            f"({artefatos['tamanho_metadados_bytes']} bytes)",
            f"  sha256 da base usada       : {artefatos['metadados']['sha256_da_base_usada']}",
        ]
    else:
        linhas.append("  Artefatos                  : NAO gravados (modo --dry-run)")
    linhas += ["", f"  AVISO: {dataset.AVISO_DADOS_SINTETICOS}"]
    return "\n".join(linhas)


def construir_analisador_de_argumentos() -> argparse.ArgumentParser:
    """Argumentos da linha de comando (executável a partir da raiz do repositório)."""
    analisador = argparse.ArgumentParser(
        prog="python -m src.cardioia.ml.train",
        description=(
            "Treina, avalia e salva o classificador de pico_risco do CardioIA (Fase 6). "
            "Os dados sao sinteticos e o modelo NAO tem validacao clinica."
        ),
    )
    analisador.add_argument(
        "--pasta-modelos",
        default=None,
        help=f"Destino dos artefatos (padrao: {PASTA_MODELOS_PADRAO}/ na raiz do projeto).",
    )
    analisador.add_argument("--semente", type=int, default=SEMENTE, help="Semente aleatoria (padrao: 42).")
    analisador.add_argument(
        "--tamanho-teste", type=float, default=TAMANHO_TESTE, help="Fracao do teste (padrao: 0.2)."
    )
    analisador.add_argument(
        "--n-estimators", type=int, default=N_ESTIMATORS_PADRAO,
        help=f"Arvores do Random Forest (padrao: {N_ESTIMATORS_PADRAO}).",
    )
    analisador.add_argument("--raiz", default=None, help="Raiz do projeto (padrao: descoberta automatica).")
    analisador.add_argument(
        "--dry-run", action="store_true", help="Mede e imprime, mas NAO grava artefatos."
    )
    analisador.add_argument(
        "--quiet", action="store_true", help="Imprime apenas o comparativo final."
    )
    return analisador


def main(argv: Sequence[str] | None = None) -> int:
    """Ponto de entrada da CLI. Retorna 0 em sucesso, 1 em falha."""
    argumentos = construir_analisador_de_argumentos().parse_args(argv)
    try:
        resultado = treinar(
            salvar=not argumentos.dry_run,
            pasta_modelos=argumentos.pasta_modelos,
            raiz=argumentos.raiz,
            semente=argumentos.semente,
            tamanho_teste=argumentos.tamanho_teste,
            n_estimators=argumentos.n_estimators,
        )
    except (dataset.ErroDeDados, ErroDeTreino, evaluate.ErroDeAvaliacao) as erro:
        print(f"[erro] {type(erro).__name__}: {erro}")
        return 1

    if not argumentos.quiet:
        for nome, avaliacao in resultado["avaliacoes"].items():
            print(evaluate.formatar_relatorio(avaliacao))
            print()
    print(formatar_resumo(resultado))
    return 0


if __name__ == "__main__":  # pragma: no cover - execucao via linha de comando
    raise SystemExit(main())