"""Avaliação dos classificadores de ``pico_risco`` (CardioIA — Fase 6).

Responsabilidades:

- calcular acurácia, precisão, recall e F1 (por classe e agregados);
- calcular a matriz de confusão e o relatório de classificação;
- dar destaque à **classe positiva** ``pico_risco = 1`` (o erro custoso é o falso
  negativo: deixar de sinalizar risco);
- devolver estruturas **serializáveis em JSON** (nada de tipos ``numpy`` ou ``NaN``);
- tratar explicitamente métricas que não podem ser calculadas (classe ausente em
  ``y_true`` ou nenhuma previsão positiva), devolvendo ``None`` + um aviso, em vez de
  um ``0.0`` silencioso que mascararia um problema.

O módulo não treina e não grava artefatos: apenas mede. Todo o cálculo vem do
``scikit-learn`` — nenhuma métrica é reimplementada.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

CLASSE_POSITIVA = 1
ROTULOS = (0, 1)
NOMES_DAS_CLASSES = ("sem_pico_de_risco", "pico_de_risco")

#: Ordem de interpretabilidade usada APENAS como último desempate do critério de
#: seleção: o primeiro da lista é o mais interpretável.
ORDEM_INTERPRETABILIDADE: tuple[str, ...] = ("Regressao Logistica", "Random Forest")

CRITERIO_DE_SELECAO = (
    "maior recall da classe pico_risco=1 no conjunto de teste; desempate por acuracia; "
    "depois por interpretabilidade"
)


class ErroDeAvaliacao(Exception):
    """Entrada inválida para avaliação (tamanhos diferentes, alvo não binário, etc.)."""


def _serializar(valor: Any) -> float | int | bool | None:
    """Converte escalares do numpy em tipos nativos; ``NaN``/``inf`` viram ``None``."""
    if valor is None:
        return None
    if isinstance(valor, (bool, np.bool_)):
        return bool(valor)
    if isinstance(valor, (int, np.integer)):
        return int(valor)
    if isinstance(valor, (float, np.floating)):
        numero = float(valor)
        if math.isnan(numero) or math.isinf(numero):
            return None
        return numero
    raise ErroDeAvaliacao(f"Valor nao numerico em metrica: {valor!r} ({type(valor).__name__}).")


def _serializar_dicionario(dados: Mapping[str, Any]) -> dict[str, Any]:
    """Serializa recursivamente o dicionário devolvido por ``classification_report``."""
    convertido: dict[str, Any] = {}
    for chave, valor in dados.items():
        if isinstance(valor, Mapping):
            convertido[str(chave)] = _serializar_dicionario(valor)
        else:
            convertido[str(chave)] = _serializar(valor)
    return convertido


def _converter_para_vetor(dados: Any, nome: str) -> np.ndarray:
    if isinstance(dados, (pd.Series, pd.DataFrame)):
        dados = dados.to_numpy()
    vetor = np.asarray(dados).ravel()
    if vetor.size == 0:
        raise ErroDeAvaliacao(f"'{nome}' esta vazio.")
    return vetor


def _validar_binario(vetor: np.ndarray, nome: str) -> None:
    valores = {int(v) for v in np.unique(vetor)}
    if not valores <= set(ROTULOS):
        raise ErroDeAvaliacao(
            f"'{nome}' deve conter apenas {sorted(ROTULOS)}; valores encontrados: {sorted(valores)}."
        )


def _media_ponderada(pares: Sequence[tuple[float | None, int]], ponderar: bool) -> float | None:
    definidos = [(valor, peso) for valor, peso in pares if valor is not None]
    if not definidos:
        return None
    if not ponderar:
        return float(sum(valor for valor, _ in definidos) / len(definidos))
    peso_total = sum(peso for _, peso in definidos)
    if peso_total == 0:
        return None
    return float(sum(valor * peso for valor, peso in definidos) / peso_total)


def avaliar_predicoes(
    y_true: Any,
    y_pred: Any,
    nome_do_modelo: str | None = None,
    nome_do_alvo: str = "pico_risco",
) -> dict[str, Any]:
    """Calcula todas as métricas de uma lista de previsões já feitas.

    Args:
        y_true: rótulos reais (0/1).
        y_pred: rótulos previstos (0/1).
        nome_do_modelo: rótulo do avaliado (aparece no resultado e no relatório).
        nome_do_alvo: nome da variável-alvo, para os textos.

    Raises:
        ErroDeAvaliacao: tamanhos diferentes, vetores vazios ou valores fora de {0, 1}.
    """
    verdadeiros = _converter_para_vetor(y_true, "y_true")
    previstos = _converter_para_vetor(y_pred, "y_pred")
    if verdadeiros.size != previstos.size:
        raise ErroDeAvaliacao(
            f"y_true e y_pred com tamanhos diferentes: {verdadeiros.size} e {previstos.size}."
        )
    _validar_binario(verdadeiros, "y_true")
    _validar_binario(previstos, "y_pred")

    avisos: list[str] = []
    suportes = {classe: int((verdadeiros == classe).sum()) for classe in ROTULOS}
    previstos_positivos = int((previstos == CLASSE_POSITIVA).sum())

    precisao = precision_score(verdadeiros, previstos, labels=list(ROTULOS), average=None, zero_division=0)
    recall = recall_score(verdadeiros, previstos, labels=list(ROTULOS), average=None, zero_division=0)
    f1 = f1_score(verdadeiros, previstos, labels=list(ROTULOS), average=None, zero_division=0)

    classes: dict[str, dict[str, Any]] = {}
    for posicao, classe in enumerate(ROTULOS):
        # Regras explícitas de indefinição (em vez de 0.0 silencioso):
        # - recall indefinido se a classe não aparece em y_true;
        # - precisão indefinida se o modelo não previu nenhum exemplo da classe;
        # - F1 indefinido em qualquer um dos dois casos.
        recall_indefinido = suportes[classe] == 0
        precisao_indefinida = int((previstos == classe).sum()) == 0
        if recall_indefinido:
            avisos.append(
                f"recall da classe {classe} indefinido: nenhum exemplo real dessa classe em y_true."
            )
        if precisao_indefinida:
            avisos.append(
                f"precisao da classe {classe} indefinida: o modelo nao previu nenhum exemplo "
                "dessa classe."
            )
        classes[str(classe)] = {
            "suporte": suportes[classe],
            "precisao": None if precisao_indefinida else _serializar(precisao[posicao]),
            "recall": None if recall_indefinido else _serializar(recall[posicao]),
            "f1": None if (recall_indefinido or precisao_indefinida) else _serializar(f1[posicao]),
        }

    pares_precisao = [(classes[str(c)]["precisao"], suportes[c]) for c in ROTULOS]
    pares_recall = [(classes[str(c)]["recall"], suportes[c]) for c in ROTULOS]
    pares_f1 = [(classes[str(c)]["f1"], suportes[c]) for c in ROTULOS]

    matriz = confusion_matrix(verdadeiros, previstos, labels=list(ROTULOS))
    tn, fp, fn, tp = (int(valor) for valor in matriz.ravel())
    nomes = [f"{nome_do_alvo}={classe}" for classe in ROTULOS]

    positivo = classes[str(CLASSE_POSITIVA)]
    negativo = classes["0"]
    return {
        "nome_do_modelo": nome_do_modelo,
        "nome_do_alvo": nome_do_alvo,
        "amostras": int(verdadeiros.size),
        "positivos_reais": suportes[CLASSE_POSITIVA],
        "positivos_previstos": previstos_positivos,
        "acuracia": _serializar(accuracy_score(verdadeiros, previstos)),
        "classe_positiva": CLASSE_POSITIVA,
        "recall_classe_1": positivo["recall"],
        "precisao_classe_1": positivo["precisao"],
        "f1_classe_1": positivo["f1"],
        "recall_classe_0": negativo["recall"],
        "precisao_classe_0": negativo["precisao"],
        "f1_classe_0": negativo["f1"],
        "classes": classes,
        "macro": {
            "precisao": _media_ponderada(pares_precisao, ponderar=False),
            "recall": _media_ponderada(pares_recall, ponderar=False),
            "f1": _media_ponderada(pares_f1, ponderar=False),
        },
        "weighted": {
            "precisao": _media_ponderada(pares_precisao, ponderar=True),
            "recall": _media_ponderada(pares_recall, ponderar=True),
            "f1": _media_ponderada(pares_f1, ponderar=True),
        },
        "matriz_confusao": [[tn, fp], [fn, tp]],
        "verdadeiros_negativos": tn,
        "falsos_positivos": fp,
        "falsos_negativos": fn,
        "verdadeiros_positivos": tp,
        "relatorio_classificacao": _serializar_dicionario(
            classification_report(
                verdadeiros, previstos, labels=list(ROTULOS), target_names=nomes,
                output_dict=True, zero_division=0,
            )
        ),
        "relatorio_texto": classification_report(
            verdadeiros, previstos, labels=list(ROTULOS), target_names=nomes, zero_division=0
        ),
        "avisos": avisos,
    }


def _extrair_predicoes(modelo: Any, X: Any) -> np.ndarray:
    """Obtém as classes previstas de um estimador treinado, com erro claro se falhar."""
    if not hasattr(modelo, "predict"):
        raise ErroDeAvaliacao(
            f"O objeto {type(modelo).__name__} nao implementa 'predict' e nao pode ser avaliado."
        )
    try:
        return _converter_para_vetor(modelo.predict(X), "previsoes")
    except Exception as erro:  # pragma: no cover - depende do estimador avaliado
        raise ErroDeAvaliacao(
            f"Falha ao prever com {type(modelo).__name__}: {type(erro).__name__}: {erro}"
        ) from erro


def avaliar_modelo(
    modelo: Any, X: Any, y: Any, nome_do_modelo: str | None = None
) -> dict[str, Any]:
    """Avalia um estimador treinado sobre ``(X, y)`` e devolve métricas serializáveis."""
    return avaliar_predicoes(
        y,
        _extrair_predicoes(modelo, X),
        nome_do_modelo=nome_do_modelo or type(modelo).__name__,
    )


def acuracia_da_regra_deterministica(y_true: Any, y_pred_regra: Any) -> float:
    """Acurácia do oráculo determinístico do material oficial (teto de referência).

    Em relação ao conjunto de teste este número é **tautologicamente perfeito**, porque a
    regra é a própria definição do rótulo. Serve para dimensionar quanto do desempenho
    dos modelos vem de aprendizado e quanto vem da construção da base sintética.
    """
    avaliacao = avaliar_predicoes(y_true, y_pred_regra, nome_do_modelo="Regra deterministica")
    return float(avaliacao["acuracia"])


def _posicao_interpretabilidade(nome: str, ordem: Sequence[str] | None = None) -> int:
    """Posição do modelo na ordem de interpretabilidade (menor = mais interpretável)."""
    referencia = list(ordem or ORDEM_INTERPRETABILIDADE)
    return referencia.index(nome) if nome in referencia else len(referencia)


def comparar_modelos(avaliacoes: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Monta o comparativo (uma linha por modelo) já ordenado pelo critério declarado.

    O critério é o mesmo do plano: maior recall da classe ``pico_risco=1``; empate é
    desempatado por acurácia e, por último, por interpretabilidade. Métricas indefinidas
    (``None``) são tratadas como piores que qualquer valor numérico.
    """
    linhas = []
    for nome, avaliacao in avaliacoes.items():
        linhas.append(
            {
                "modelo": str(nome),
                "acuracia": avaliacao.get("acuracia"),
                "recall_classe_1": avaliacao.get("recall_classe_1"),
                "precisao_classe_1": avaliacao.get("precisao_classe_1"),
                "f1_classe_1": avaliacao.get("f1_classe_1"),
                "falsos_negativos": avaliacao.get("falsos_negativos"),
                "falsos_positivos": avaliacao.get("falsos_positivos"),
                "interpretabilidade": _posicao_interpretabilidade(str(nome)),
            }
        )
    return ordenar_por_criterio(linhas)


def ordenar_por_criterio(linhas: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Ordena o comparativo pelo critério pré-declarado (não ajusta nada no teste)."""

    def chave(linha: Mapping[str, Any]) -> tuple[float, float, int, str]:
        recall = linha.get("recall_classe_1")
        acuracia = linha.get("acuracia")
        return (
            -float(recall) if recall is not None else float("inf"),
            -float(acuracia) if acuracia is not None else float("inf"),
            int(linha.get("interpretabilidade", 99)),
            str(linha.get("modelo", "")),
        )

    return [dict(linha) for linha in sorted(linhas, key=chave)]


def formatar_relatorio(avaliacao: Mapping[str, Any]) -> str:
    """Texto pronto para CLI/notebook, com avisos explícitos de métricas indefinidas."""
    matriz = avaliacao["matriz_confusao"]
    linhas = [
        "=" * 72,
        str(avaliacao.get("nome_do_modelo") or "modelo"),
        "=" * 72,
        f"Acuracia: {avaliacao['acuracia']}",
        "",
        "Matriz de confusao (linhas = real, colunas = previsto, ordem [0, 1]):",
        pd.DataFrame(
            matriz,
            index=["real pico_risco=0", "real pico_risco=1"],
            columns=["previsto 0", "previsto 1"],
        ).to_string(),
        "",
        "Relatorio de classificacao:",
        str(avaliacao["relatorio_texto"]).rstrip(),
        (
            "Leitura: TN = {tn} | FP = {fp} | FN = {fn} | TP = {tp}".format(
                tn=avaliacao["verdadeiros_negativos"],
                fp=avaliacao["falsos_positivos"],
                fn=avaliacao["falsos_negativos"],
                tp=avaliacao["verdadeiros_positivos"],
            )
        ),
    ]
    for aviso in avaliacao.get("avisos", []):
        linhas.append(f"[aviso] {aviso}")
    return "\n".join(linhas)