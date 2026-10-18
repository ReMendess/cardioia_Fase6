"""Inferência do modelo salvo de ``pico_risco`` (CardioIA — Fase 6).

Este módulo é a **interface de uso do modelo** e foi desenhado para ser chamado tanto de
um notebook quanto de uma *tool* do OpenAI Agents SDK na próxima etapa:

    from cardioia.ml import predict

    resultado = predict.prever_pico_risco({
        "idade_media": 62, "carga_sistema": 92, "tempo_espera": 75, "fc_media": 118,
        "spo2_media": 89, "pa_sistolica": 100, "temperatura": 37.5,
        "densidade_eventos": 7, "eventos_24h": 5, "disponibilidade_recursos": 25,
    })

Contrato:

- todas as features exigidas pelo modelo devem estar presentes;
- as colunas são **reordenadas** conforme a ordem gravada no metadata (a ordem de entrada
  do chamador não importa);
- campos desconhecidos, valores não numéricos, ``NaN``/``inf`` e valores negativos são
  **rejeitados** com erro explícito;
- valores fora das faixas observadas no treino geram **aviso** (não erro);
- se o artefato não existir, o erro traz o comando exato para treinar. **Nunca** treinamos
  silenciosamente durante uma consulta, e nenhuma probabilidade é inventada.

A probabilidade devolvida é o valor **cru** do classificador (sem arredondamento prévio).
O arredondamento é responsabilidade de quem apresenta o resultado.

Aviso de uso: modelo treinado em dados **sintéticos**. Não é dispositivo médico e não tem
validação clínica.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

from ..config import (
    COMANDO_DE_TREINO,
    LIMIAR_DE_DECISAO,
    NOME_ARQUIVO_METADADOS,
    NOME_ARQUIVO_MODELO,
    PASTA_MODELOS_PADRAO,
)
from . import dataset
from .dataset import COLUNAS_PREDITORAS, VARIAVEL_ALVO

# Reexportados de ``cardioia.config`` (fonte única); mantidos aqui porque são parte da
# interface pública já usada pelo notebook, pelos testes e pelo CLI.
__all__ = [
    "COMANDO_DE_TREINO",
    "LIMIAR_DE_DECISAO",
    "NOME_ARQUIVO_METADADOS",
    "NOME_ARQUIVO_MODELO",
    "PASTA_MODELOS_PADRAO",
]

CLASSE_POSITIVA = 1
ROTULOS: Mapping[int, str] = {0: "sem_pico_de_risco", 1: "pico_de_risco"}

AVISO_DE_USO = (
    "Modelo treinado em dados SINTETICOS, sem validacao clinica. Este resultado nao e "
    "diagnostico, nao e prescricao e nao substitui avaliacao profissional de saude."
)


class ErroDePrevisao(Exception):
    """Erro base da camada de inferência."""


class ModeloNaoEncontradoError(ErroDePrevisao):
    """O artefato do modelo (ou o metadata) não existe no caminho esperado."""


class PacienteInvalidoError(ErroDePrevisao):
    """O caso recebido viola o contrato de entrada do modelo."""


#: Mapa reverso das correções documentadas do material oficial: nome esperado ->
#: grafia usada no arquivo da aula. Serve para produzir mensagens de erro úteis.
_ALIAS_DE_ENTRADA: Mapping[str, str] = {
    esperado: grafia_usada for grafia_usada, esperado in dataset.RENOMEAR_COLUNAS_OFICIAIS.items()
}


# --------------------------------------------------------------------------- #
# Caminhos e carga dos artefatos
# --------------------------------------------------------------------------- #


def pasta_de_modelos_padrao(raiz: str | Path | None = None) -> Path:
    """Pasta de artefatos. Padrão: ``models/`` na raiz do projeto (sem caminho absoluto)."""
    raiz_resolvida = Path(raiz) if raiz is not None else dataset.localizar_raiz_do_projeto()
    return (raiz_resolvida or Path.cwd()) / PASTA_MODELOS_PADRAO


def _resolver_caminho(
    caminho: str | Path | None, pasta_modelos: str | Path | None, raiz: str | Path | None, nome: str
) -> Path:
    if caminho is not None:
        return Path(caminho)
    pasta = Path(pasta_modelos) if pasta_modelos is not None else pasta_de_modelos_padrao(raiz)
    return pasta / nome


def carregar_metadados(
    caminho: str | Path | None = None,
    pasta_modelos: str | Path | None = None,
    raiz: str | Path | None = None,
) -> dict[str, Any]:
    """Lê o JSON de metadados do modelo treinado.

    Raises:
        ModeloNaoEncontradoError: se o arquivo não existir (com instrução de treino).
    """
    destino = _resolver_caminho(caminho, pasta_modelos, raiz, NOME_ARQUIVO_METADADOS)
    if not destino.is_file():
        raise ModeloNaoEncontradoError(
            f"Metadados nao encontrados em '{destino}'. Treine o modelo antes de consultar:\n"
            f"    {COMANDO_DE_TREINO}\n"
            "Ou informe o caminho correto em 'caminho=' / 'pasta_modelos='."
        )
    try:
        return json.loads(destino.read_text(encoding="utf-8"))
    except json.JSONDecodeError as erro:
        raise ModeloNaoEncontradoError(
            f"Metadados invalidos em '{destino}': {erro}. Gere novamente com '{COMANDO_DE_TREINO}'."
        ) from erro


def carregar_modelo(
    caminho: str | Path | None = None,
    pasta_modelos: str | Path | None = None,
    raiz: str | Path | None = None,
) -> Any:
    """Carrega o modelo salvo com ``joblib``.

    Raises:
        ModeloNaoEncontradoError: se o artefato não existir. **Nada é treinado aqui.**
    """
    destino = _resolver_caminho(caminho, pasta_modelos, raiz, NOME_ARQUIVO_MODELO)
    if not destino.is_file():
        raise ModeloNaoEncontradoError(
            f"Modelo nao encontrado em '{destino}'. Treine o modelo antes de consultar:\n"
            f"    {COMANDO_DE_TREINO}\n"
            "Ou informe o caminho correto em 'caminho=' / 'pasta_modelos='."
        )
    if not hasattr(joblib.load, "__call__"):  # pragma: no cover - salvaguarda defensiva
        raise ErroDePrevisao("joblib indisponivel para carregar o modelo.")
    return joblib.load(destino)


def features_do_modelo(
    metadados: Mapping[str, Any] | None = None, modelo: Any | None = None
) -> list[str]:
    """Descobre o contrato de entrada: features (e ordem) do metadata ou do estimador."""
    if metadados:
        features = metadados.get("features")
        if isinstance(features, Sequence) and features:
            return [str(coluna) for coluna in features]
    if modelo is not None:
        nomes = getattr(modelo, "feature_names_in_", None)
        if nomes is not None:
            return [str(coluna) for coluna in nomes]
    return list(COLUNAS_PREDITORAS)


# --------------------------------------------------------------------------- #
# Validação da entrada
# --------------------------------------------------------------------------- #


def _extrair_dicionario(paciente: Any) -> dict[str, Any]:
    """Aceita ``dict``/``Mapping`` ou ``DataFrame`` de uma linha e devolve um dicionário."""
    if isinstance(paciente, pd.DataFrame):
        if len(paciente) != 1:
            raise PacienteInvalidoError(
                f"Esperado um DataFrame com exatamente 1 linha; recebido {len(paciente)}. "
                "Use 'prever_varios' para lotes."
            )
        return {str(coluna): valor for coluna, valor in paciente.iloc[0].to_dict().items()}
    if isinstance(paciente, Mapping):
        return {str(coluna): valor for coluna, valor in paciente.items()}
    raise PacienteInvalidoError(
        f"Entrada deve ser um dict ou um DataFrame de 1 linha; recebido {type(paciente).__name__}."
    )


def validar_paciente(
    paciente: Any,
    features: Sequence[str] | None = None,
    faixas: Mapping[str, Sequence[float]] | None = None,
    permitir_campos_extras: bool = False,
) -> tuple[dict[str, float], list[str]]:
    """Valida um caso e devolve ``(valores_na_ordem_do_modelo, avisos)``.

    Rejeita (levantando :class:`PacienteInvalidoError`): campos ausentes, campos
    desconhecidos, valores não numéricos ou booleanos, ``NaN``/``inf`` e valores
    negativos. Valores fora da faixa observada no treino geram **aviso**.

    Args:
        paciente: ``dict`` ou ``DataFrame`` de 1 linha.
        features: contrato de entrada (ordem esperada pelo modelo).
        faixas: faixas por coluna (do metadata). Padrão: faixas de projeto do gerador.
        permitir_campos_extras: se ``True``, ignora campos desconhecidos em vez de falhar.
    """
    dados = _extrair_dicionario(paciente)
    esperadas = [str(coluna) for coluna in (features or COLUNAS_PREDITORAS)]

    ausentes = [coluna for coluna in esperadas if coluna not in dados]
    if ausentes:
        # Caso real do material da aula: a coluna chega grafada com espaço, então a
        # coluna esperada fica "ausente". A mensagem precisa dizer isso explicitamente.
        dicas = [
            f"'{alias}' (com espaco) foi informado no lugar de '{coluna}'"
            for coluna in ausentes
            for alias in [_ALIAS_DE_ENTRADA.get(coluna)]
            if alias and alias in dados
        ]
        sufixo = (" Dica: " + "; ".join(dicas) + ".") if dicas else ""
        raise PacienteInvalidoError(
            "Campos obrigatorios ausentes: " + ", ".join(ausentes)
            + ". O modelo espera exatamente: " + ", ".join(esperadas) + "." + sufixo
        )

    desconhecidos = [coluna for coluna in dados if coluna not in esperadas]
    if desconhecidos and not permitir_campos_extras:
        dica = ""
        if any(" " in coluna for coluna in desconhecidos):
            dica = (
                " Atencao: o material da aula grafa 'disponibilidade recursos' (com espaco); "
                "o contrato do modelo exige 'disponibilidade_recursos'."
            )
        raise PacienteInvalidoError(
            "Campos desconhecidos no caso: " + ", ".join(sorted(desconhecidos))
            + ". O modelo espera exatamente: " + ", ".join(esperadas) + "." + dica
        )

    valores: dict[str, float] = {}
    avisos: list[str] = []
    faixas_efetivas: dict[str, Sequence[float]] = dict(dataset.FAIXAS_ESPERADAS)
    if faixas:
        faixas_efetivas.update({str(coluna): faixa for coluna, faixa in faixas.items()})

    for coluna in esperadas:
        valor = dados[coluna]
        if isinstance(valor, (bool, np.bool_)):
            raise PacienteInvalidoError(
                f"'{coluna}' recebeu um booleano ({valor!r}); informe um valor numerico."
            )
        if isinstance(valor, str):
            raise PacienteInvalidoError(
                f"'{coluna}' recebeu texto ({valor!r}); informe um valor numerico "
                "(a conversao nao e feita silenciosamente)."
            )
        if not isinstance(valor, (int, float, np.integer, np.floating)):
            raise PacienteInvalidoError(
                f"'{coluna}' recebeu {type(valor).__name__} ({valor!r}); esperado numero."
            )
        numero = float(valor)
        if math.isnan(numero) or math.isinf(numero):
            raise PacienteInvalidoError(f"'{coluna}' recebeu um valor invalido ({valor!r}).")
        if numero < 0:
            raise PacienteInvalidoError(
                f"'{coluna}' recebeu valor negativo ({numero}); as features do gerador sao "
                "nao negativas."
            )
        valores[coluna] = numero

        faixa = faixas_efetivas.get(coluna)
        if faixa is not None and len(faixa) == 2:
            minimo, maximo = float(faixa[0]), float(faixa[1])
            if numero < minimo or numero > maximo:
                avisos.append(
                    f"valor fora da faixa observada no treino para '{coluna}': {numero} "
                    f"(faixa {minimo} a {maximo}; base sintetica do material oficial)."
                )

    return valores, avisos


# --------------------------------------------------------------------------- #
# Previsão
# --------------------------------------------------------------------------- #


def _probabilidade_da_classe_positiva(
    modelo: Any, entrada: pd.DataFrame
) -> tuple[float | None, list[str]]:
    """Probabilidade da classe 1 (valor cru), com avisos quando não é calculável."""
    if not hasattr(modelo, "predict_proba"):
        return None, [
            f"o classificador {type(modelo).__name__} nao implementa 'predict_proba'; "
            "a probabilidade nao foi informada."
        ]
    probabilidades = np.asarray(modelo.predict_proba(entrada))[0]
    classes = list(getattr(modelo, "classes_", [0, 1]))
    if CLASSE_POSITIVA not in classes:
        return None, [
            f"a classe positiva {CLASSE_POSITIVA} nao existe no modelo treinado "
            f"(classes: {classes}); a probabilidade nao foi informada."
        ]
    return float(probabilidades[classes.index(CLASSE_POSITIVA)]), []


def prever_pico_risco(
    paciente: Any,
    caminho_modelo: str | Path | None = None,
    pasta_modelos: str | Path | None = None,
    raiz: str | Path | None = None,
    limiar: float = LIMIAR_DE_DECISAO,
    modelo: Any | None = None,
    metadados: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Prevê ``pico_risco`` para um caso, devolvendo um resultado estruturado.

    Args:
        paciente: ``dict`` ou ``DataFrame`` de 1 linha com as features do modelo.
        caminho_modelo: caminho do ``.pkl``. Padrão: ``models/modelo_pico_risco.pkl``.
        pasta_modelos: pasta alternativa dos artefatos.
        raiz: raiz do projeto (descoberta automaticamente quando ``None``).
        limiar: limiar explícito de decisão (padrão 0.5, operacional — não clínico).
        modelo: estimador já carregado (evita releitura em lote e nos testes).
        metadados: metadata já carregado.

    Returns:
        Dicionário com ``pico_risco_previsto``, ``probabilidade_risco`` (crua),
        ``classificacao_risco``, ``modelo_utilizado``, ``limiar_decisao``,
        ``features_utilizadas``, ``entrada``, ``origem_do_modelo``, ``avisos`` e
        ``aviso_de_uso``.

    Raises:
        ModeloNaoEncontradoError: artefato ausente (com o comando de treino).
        PacienteInvalidoError: entrada fora do contrato do modelo.
        ErroDePrevisao: limiar inválido ou falha do estimador.
    """
    limite = float(limiar)
    if not 0.0 <= limite <= 1.0:
        raise ErroDePrevisao(f"'limiar' deve estar entre 0 e 1; recebido {limiar!r}.")

    if metadados is None:
        metadados = carregar_metadados(pasta_modelos=pasta_modelos, raiz=raiz)
    if modelo is None:
        modelo = carregar_modelo(caminho_modelo, pasta_modelos=pasta_modelos, raiz=raiz)

    features = features_do_modelo(metadados, modelo)
    valores, avisos = validar_paciente(
        paciente, features=features, faixas=metadados.get("faixas_esperadas")
    )
    entrada = pd.DataFrame([valores])[features]  # reordenacao garantida pelo contrato

    try:
        previsto_pelo_estimador = int(modelo.predict(entrada)[0])
    except Exception as erro:  # pragma: no cover - depende do estimador
        raise ErroDePrevisao(f"Falha ao prever: {type(erro).__name__}: {erro}") from erro

    probabilidade, avisos_probabilidade = _probabilidade_da_classe_positiva(modelo, entrada)
    avisos.extend(avisos_probabilidade)

    if probabilidade is None:
        # Sem probabilidade, a decisão segue o estimador (comportamento documentado).
        decisao = previsto_pelo_estimador
        regra = "classe prevista pelo estimador (probabilidade indisponivel)"
    else:
        decisao = CLASSE_POSITIVA if probabilidade >= limite else 0
        regra = f"probabilidade_risco >= limiar ({limite})"

    return {
        "pico_risco_previsto": decisao,
        "probabilidade_risco": probabilidade,
        "classificacao_risco": ROTULOS[int(decisao)],
        "modelo_utilizado": str(metadados.get("nome_do_modelo") or type(modelo).__name__),
        "classe_do_estimador": str(metadados.get("classe_do_estimador") or type(modelo).__name__),
        "variavel_alvo": str(metadados.get("variavel_alvo") or VARIAVEL_ALVO),
        "limiar_decisao": limite,
        "regra_de_decisao": regra,
        "features_utilizadas": features,
        "entrada": valores,
        "classe_prevista_pelo_estimador": previsto_pelo_estimador,
        "origem_do_modelo": {
            "caminho_modelo": str(
                _resolver_caminho(caminho_modelo, pasta_modelos, raiz, NOME_ARQUIVO_MODELO)
            ),
            "semente": metadados.get("semente"),
            "sha256_da_base_usada": metadados.get("sha256_da_base_usada"),
            "origem_da_base": metadados.get("origem_da_base"),
        },
        "avisos": avisos,
        "aviso_de_uso": AVISO_DE_USO,
    }


def prever_varios(
    pacientes: Sequence[Any],
    caminho_modelo: str | Path | None = None,
    pasta_modelos: str | Path | None = None,
    raiz: str | Path | None = None,
    limiar: float = LIMIAR_DE_DECISAO,
) -> list[dict[str, Any]]:
    """Aplica :func:`prever_pico_risco` a uma lista, carregando os artefatos uma única vez."""
    metadados = carregar_metadados(pasta_modelos=pasta_modelos, raiz=raiz)
    modelo = carregar_modelo(caminho_modelo, pasta_modelos=pasta_modelos, raiz=raiz)
    return [
        prever_pico_risco(
            caso,
            caminho_modelo=caminho_modelo,
            pasta_modelos=pasta_modelos,
            raiz=raiz,
            limiar=limiar,
            modelo=modelo,
            metadados=metadados,
        )
        for caso in pacientes
    ]


def resumo_para_agente(resultado: Mapping[str, Any]) -> str:
    """Uma linha legível, pronta para ser devolvida por uma *tool* de agente."""
    probabilidade = resultado.get("probabilidade_risco")
    texto = f"{probabilidade:.4f}" if isinstance(probabilidade, float) else "indisponivel"
    return (
        f"pico_risco_previsto={resultado.get('pico_risco_previsto')} | "
        f"probabilidade_risco={texto} | "
        f"classificacao_risco={resultado.get('classificacao_risco')} | "
        f"modelo={resultado.get('modelo_utilizado')} | "
        f"limiar={resultado.get('limiar_decisao')} | "
        "base=sintetica_sem_validacao_clinica"
    )