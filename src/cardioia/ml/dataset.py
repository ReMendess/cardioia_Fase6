"""Carregamento, validação e preparação da base sintética do CardioIA (Fase 6).

Este módulo é a **autoridade única** sobre o material oficial da aula:

- executa ``data/base_dados.py`` sem editar nem duplicar silenciosamente suas regras;
- valida colunas, tipos, valores ausentes e o contrato da variável-alvo ``pico_risco``;
- separa features e alvo garantindo que colunas derivadas do rótulo (por exemplo
  ``risk_score``) nunca entrem em ``X`` — isso seria vazamento de alvo;
- reproduz, em Python determinístico, a regra do material oficial que gera o rótulo;
- lê o caso de simulação ``data/novo_paciente.py`` corrigindo a divergência de nome
  de coluna documentada na Etapa 1.

Caminhos: resolvidos relativamente à **raiz do projeto** (a pasta que contém
``data/base_dados.py``), nunca por caminho absoluto da máquina.

Importação (três formas, todas suportadas):

    from cardioia.ml import dataset        # com src/ no sys.path
    import cardioia.ml.dataset             # idem
    python -m src.cardioia.ml.train        # executado a partir da raiz do repositório
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------- #
# Constantes de domínio (idênticas às do notebook e ao material oficial)
# --------------------------------------------------------------------------- #

VARIAVEL_ALVO = "pico_risco"

#: Ordem exata das colunas preditoras geradas por ``data/base_dados.py``.
COLUNAS_PREDITORAS: tuple[str, ...] = (
    "idade_media",
    "carga_sistema",
    "tempo_espera",
    "fc_media",
    "spo2_media",
    "pa_sistolica",
    "temperatura",
    "densidade_eventos",
    "eventos_24h",
    "disponibilidade_recursos",
)

#: Colunas derivadas do alvo que jamais podem ser usadas como feature.
COLUNAS_AUXILIARES_IGNORADAS: tuple[str, ...] = ("risk_score",)

SEMENTE = 42
TAMANHO_TESTE = 0.20

#: Limiares do rótulo, exatamente como no arquivo oficial. NÃO alterar.
LIMIARES_DO_ROTULO: Mapping[str, float] = {
    "carga_sistema": 80,             # > 80
    "tempo_espera": 60,              # > 60
    "spo2_media": 92,                # < 92
    "fc_media": 110,                 # > 110
    "eventos_24h": 4,                # > 4
    "disponibilidade_recursos": 40,  # < 40
}
CORTE_RISK_SCORE = 2

PASTA_DADOS = "data"
NOME_ARQUIVO_BASE = "base_dados.py"
NOME_ARQUIVO_CONHECIMENTO = "base_conhecimento.py"
NOME_ARQUIVO_NOVO_PACIENTE = "novo_paciente.py"

#: Faixas de projeto do gerador (usadas apenas para AVISAR sobre valores atípicos).
#: Para as contínuas, são os limites de ``clip`` do arquivo oficial; para as de
#: contagem (Poisson, sem clip), um limite prático documentado.
FAIXAS_ESPERADAS: Mapping[str, tuple[float, float]] = {
    "idade_media": (18.0, 90.0),
    "carga_sistema": (30.0, 100.0),
    "tempo_espera": (5.0, 180.0),
    "fc_media": (50.0, 150.0),
    "spo2_media": (70.0, 100.0),
    "pa_sistolica": (80.0, 200.0),
    "temperatura": (10.0, 40.0),
    "densidade_eventos": (0.0, 25.0),
    "eventos_24h": (0.0, 25.0),
    "disponibilidade_recursos": (20.0, 100.0),
}

#: Divergência conhecida do material: o arquivo da aula escreve a última coluna com
#: espaço. A correção é feita de forma explícita e registrada (nada é renomeado no
#: arquivo original).
RENOMEAR_COLUNAS_OFICIAIS: Mapping[str, str] = {
    "disponibilidade recursos": "disponibilidade_recursos",
}

AVISO_DADOS_SINTETICOS = (
    "Base sintetica gerada por regras (np.random.seed(42)). Nao ha validacao "
    "clinica: os limiares sao de projeto e nao representam conduta medica."
)


# --------------------------------------------------------------------------- #
# Erros próprios (mensagens claras > exceções genéricas)
# --------------------------------------------------------------------------- #


class ErroDeDados(Exception):
    """Erro base da camada de dados."""


class ArquivoOficialNaoEncontradoError(ErroDeDados):
    """O arquivo oficial da aula não foi encontrado no projeto."""


class BaseInvalidaError(ErroDeDados):
    """A base carregada viola o contrato esperado (colunas, tipos, nulos, alvo)."""


# --------------------------------------------------------------------------- #
# Resolução de caminhos
# --------------------------------------------------------------------------- #


def localizar_raiz_do_projeto(inicio: str | Path | None = None) -> Path | None:
    """Procura a raiz do projeto subindo os diretórios a partir de ``inicio``.

    A raiz é a pasta que contém ``data/base_dados.py``. Nenhum caminho absoluto é
    usado: o módulo funciona localmente e no Google Colab.

    Args:
        inicio: ponto de partida da busca. Padrão: diretório atual.

    Returns:
        ``Path`` da raiz ou ``None`` se não encontrada.
    """
    base = Path(inicio or Path.cwd()).resolve()
    for candidato in (base, *base.parents):
        if (candidato / PASTA_DADOS / NOME_ARQUIVO_BASE).is_file():
            return candidato
    return None


def caminho_da_base_oficial(raiz: str | Path | None = None) -> Path:
    """Retorna o caminho de ``data/base_dados.py``, ou levanta erro com instruções."""
    raiz_resolvida = Path(raiz) if raiz is not None else localizar_raiz_do_projeto()
    if raiz_resolvida is None:
        raise ArquivoOficialNaoEncontradoError(
            "Nao foi possivel localizar a raiz do projeto (pasta que contem "
            f"'{PASTA_DADOS}/{NOME_ARQUIVO_BASE}'). Execute o codigo a partir da raiz do "
            "repositorio, passe 'raiz=' explicitamente ou, no Google Colab, envie a pasta "
            "do projeto (upload do .zip) antes de rodar."
        )
    caminho = Path(raiz_resolvida) / PASTA_DADOS / NOME_ARQUIVO_BASE
    if not caminho.is_file():
        raise ArquivoOficialNaoEncontradoError(
            f"Arquivo oficial nao encontrado em '{caminho}'. Confirme que a pasta "
            f"'{PASTA_DADOS}/' do repositorio esta presente. Este modulo NAO usa replica "
            "inline do gerador: passe 'codigo_fonte=' apenas em um failover documentado."
        )
    return caminho


def ler_arquivo_oficial(nome_do_arquivo: str, raiz: str | Path | None = None) -> str:
    """Lê um dos arquivos oficiais de ``data/`` como texto (sem executar nada)."""
    raiz_resolvida = Path(raiz) if raiz is not None else localizar_raiz_do_projeto()
    if raiz_resolvida is None:
        raise ArquivoOficialNaoEncontradoError(
            "Raiz do projeto nao encontrada. Passe 'raiz=' com a pasta do repositorio."
        )
    caminho = Path(raiz_resolvida) / PASTA_DADOS / nome_do_arquivo
    if not caminho.is_file():
        raise ArquivoOficialNaoEncontradoError(f"Arquivo oficial ausente: '{caminho}'.")
    return caminho.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Carregamento da base oficial
# --------------------------------------------------------------------------- #


def _extrair_dataframe(codigo_fonte: str, origem: str) -> pd.DataFrame:
    """Executa o gerador oficial em namespace isolado e devolve o DataFrame ``df``."""
    namespace: dict[str, Any] = {}
    try:
        exec(compile(codigo_fonte, origem, "exec"), namespace)  # noqa: S102 - arquivo oficial
    except Exception as erro:  # pragma: no cover - erro de terceiros
        raise BaseInvalidaError(
            f"Falha ao executar o gerador oficial '{origem}': {type(erro).__name__}: {erro}"
        ) from erro
    if "df" not in namespace:
        raise BaseInvalidaError(
            f"O arquivo '{origem}' nao define a variavel 'df'. Esperado um DataFrame "
            "com as 10 features e a coluna 'pico_risco'."
        )
    tabela = namespace["df"]
    if not isinstance(tabela, pd.DataFrame):
        raise BaseInvalidaError(
            f"O arquivo '{origem}' produziu '{type(tabela).__name__}', esperado pandas.DataFrame."
        )
    return tabela


def carregar_base_oficial(
    raiz: str | Path | None = None,
    codigo_fonte: str | None = None,
    validar: bool = True,
    retornar_origem: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    """Carrega a base sintética executando o gerador oficial da aula.

    A semente global é fixada antes da execução, portanto chamadas repetidas produzem
    exatamente o mesmo DataFrame (reprodutibilidade verificada nos testes).

    Args:
        raiz: raiz do projeto. Padrão: descoberta automaticamente.
        codigo_fonte: código do gerador a executar. Use **somente** em um failover
            documentado (por exemplo, o notebook, quando o arquivo oficial não está
            disponível no Colab). Se ``None``, lê ``data/base_dados.py``; se o arquivo
            não existir, levanta :class:`ArquivoOficialNaoEncontradoError`.
        validar: se ``True``, aplica :func:`validar_base` e levanta
            :class:`BaseInvalidaError` em caso de inconsistência.
        retornar_origem: se ``True``, devolve ``(df, origem)`` com a rastreabilidade
            da carga (usada no metadata do modelo).

    Returns:
        ``DataFrame`` da base; ou ``(DataFrame, dict)`` quando ``retornar_origem=True``.
    """
    if codigo_fonte is None:
        caminho = caminho_da_base_oficial(raiz)
        codigo_fonte = caminho.read_text(encoding="utf-8")
        origem = f"arquivo oficial {PASTA_DADOS}/{NOME_ARQUIVO_BASE}"
        usado_failover = False
    else:
        origem = "codigo_fonte fornecido explicitamente (failover documentado)"
        usado_failover = True

    np.random.seed(SEMENTE)  # o material oficial também fixa a semente (42)
    df = _extrair_dataframe(codigo_fonte, origem)

    if validar:
        validar_base(df)

    if retornar_origem:
        return df, {
            "origem": origem,
            "usado_failover": usado_failover,
            "aviso_dados_sinteticos": AVISO_DADOS_SINTETICOS,
        }
    return df


# --------------------------------------------------------------------------- #
# Validação
# --------------------------------------------------------------------------- #


def _colunas_ausentes(df: pd.DataFrame) -> list[str]:
    exigidas = (*COLUNAS_PREDITORAS, VARIAVEL_ALVO)
    return [coluna for coluna in exigidas if coluna not in df.columns]


def validar_base(df: pd.DataFrame, exigir_alvo: bool = True) -> dict[str, Any]:
    """Valida a base e devolve um resumo serializável em JSON.

    Levanta :class:`BaseInvalidaError` quando o contrato é violado: base vazia,
    colunas ausentes, tipos não numéricos, valores ausentes ou alvo não binário.
    Duplicatas e valores fora das faixas do gerador são reportados como *avisos*.
    """
    if not isinstance(df, pd.DataFrame):
        raise BaseInvalidaError(f"Esperado pandas.DataFrame, recebido {type(df).__name__}.")
    if df.empty:
        raise BaseInvalidaError("A base esta vazia (0 linhas).")

    if exigir_alvo:
        ausentes = _colunas_ausentes(df)
    else:
        ausentes = [coluna for coluna in COLUNAS_PREDITORAS if coluna not in df.columns]
    if ausentes:
        raise BaseInvalidaError(
            "Colunas obrigatorias ausentes: " + ", ".join(ausentes) + ". Esperado: "
            + ", ".join((*COLUNAS_PREDITORAS, VARIAVEL_ALVO)) + "."
        )

    avaliadas = [*COLUNAS_PREDITORAS, *([VARIAVEL_ALVO] if exigir_alvo else [])]
    nao_numericas = [
        coluna
        for coluna in avaliadas
        if not pd.api.types.is_numeric_dtype(df[coluna]) or pd.api.types.is_bool_dtype(df[coluna])
    ]
    if nao_numericas:
        raise BaseInvalidaError(
            "Colunas devem ser numericas: " + ", ".join(nao_numericas)
            + ". Tipos encontrados: "
            + ", ".join(f"{coluna}={df[coluna].dtype}" for coluna in nao_numericas)
        )

    nulos_por_coluna = {coluna: int(df[coluna].isna().sum()) for coluna in avaliadas}
    nulos_total = int(sum(nulos_por_coluna.values()))
    if nulos_total:
        detalhe = ", ".join(f"{c}={n}" for c, n in nulos_por_coluna.items() if n)
        raise BaseInvalidaError(
            f"A base possui {nulos_total} valores ausentes ({detalhe}). "
            "O fluxo do CardioIA nao imputa dados: corrija a origem."
        )

    alvo_valores: list[int] = []
    if exigir_alvo:
        alvo_valores = sorted(int(v) for v in pd.unique(df[VARIAVEL_ALVO]))
        if alvo_valores != [0, 1]:
            raise BaseInvalidaError(
                f"'{VARIAVEL_ALVO}' deve ser binaria {{0, 1}}; valores encontrados: {alvo_valores}."
            )

    avisos: list[str] = []
    duplicadas = int(df.duplicated().sum())
    if duplicadas:
        avisos.append(f"Existem {duplicadas} linhas duplicadas na base.")

    fora_da_faixa: dict[str, int] = {}
    for coluna, (minimo, maximo) in FAIXAS_ESPERADAS.items():
        if coluna in df.columns:
            quantidade = int(((df[coluna] < minimo) | (df[coluna] > maximo)).sum())
            if quantidade:
                fora_da_faixa[coluna] = quantidade
    if fora_da_faixa:
        avisos.append(
            "Valores fora das faixas de projeto do gerador: "
            + ", ".join(f"{coluna}={quantidade}" for coluna, quantidade in fora_da_faixa.items())
        )

    return {
        "dimensoes": {"linhas": int(df.shape[0]), "colunas": int(df.shape[1])},
        "colunas": [str(coluna) for coluna in df.columns],
        "colunas_preditotas_presentes": list(COLUNAS_PREDITORAS),
        "tipos": {coluna: str(df[coluna].dtype) for coluna in df.columns},
        "valores_ausentes_total": nulos_total,
        "valores_ausentes_por_coluna": {c: n for c, n in nulos_por_coluna.items() if n},
        "linhas_duplicadas": duplicadas,
        "valores_fora_da_faixa": fora_da_faixa,
        "alvo_presente": bool(exigir_alvo),
        "alvo_valores": alvo_valores,
        "alvo_proporcao_positiva": (float(df[VARIAVEL_ALVO].mean()) if exigir_alvo else None),
        "avisos": avisos,
    }


def faixas_observadas(df: pd.DataFrame) -> dict[str, list[float]]:
    """Faixas (mínimo, máximo) realmente observadas na base, por feature.

    Vão para o metadata do modelo e permitem que ``predict`` avise sobre valores
    atípicos usando a base de treino, em vez de constantes fixas no código.
    """
    return {
        coluna: [float(df[coluna].min()), float(df[coluna].max())]
        for coluna in COLUNAS_PREDITORAS
        if coluna in df.columns
    }


# --------------------------------------------------------------------------- #
# Regra determinística do material oficial
# --------------------------------------------------------------------------- #

#: (rótulo textual, coluna, comparador, limiar) — espelha o arquivo oficial.
_INDICADORES: tuple[tuple[str, str, Any, float], ...] = (
    ("carga_sistema > 80", "carga_sistema", lambda v, l: v > l, 80),
    ("tempo_espera > 60", "tempo_espera", lambda v, l: v > l, 60),
    ("spo2_media < 92", "spo2_media", lambda v, l: v < l, 92),
    ("fc_media > 110", "fc_media", lambda v, l: v > l, 110),
    ("eventos_24h > 4", "eventos_24h", lambda v, l: v > l, 4),
    ("disponibilidade_recursos < 40", "disponibilidade_recursos", lambda v, l: v < l, 40),
)


def calcular_risk_score(df: pd.DataFrame) -> pd.Series:
    """Recalcula, em Python determinístico, o ``risk_score`` do arquivo oficial.

    É a soma dos 6 indicadores booleanos. Este valor **não** é feature do modelo:
    serve apenas como verificação/oráculo determinístico.
    """
    faltando = [chave for _, chave, _, _ in _INDICADORES if chave not in df.columns]
    if faltando:
        raise BaseInvalidaError(
            "Nao e possivel calcular o risk_score: colunas ausentes " + ", ".join(faltando) + "."
        )
    total = pd.Series(0, index=df.index, dtype=int)
    for _, chave, comparador, limiar in _INDICADORES:
        total = total + comparador(df[chave], limiar).astype(int)
    return total


def indicadores_ativos(caso: Mapping[str, Any]) -> list[str]:
    """Lista, em texto, quais indicadores do rótulo oficial disparam para um caso."""
    return [
        rotulo
        for rotulo, chave, comparador, limiar in _INDICADORES
        if comparador(float(caso[chave]), float(limiar))
    ]


def rotulo_pela_regra(df: pd.DataFrame) -> pd.Series:
    """Aplica a regra oficial (``risk_score >= 2``) e devolve o ``pico_risco`` esperado."""
    return (calcular_risk_score(df) >= CORTE_RISK_SCORE).astype(int)


def separar_features_e_alvo(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa ``X`` (10 features originais) e ``y`` (``pico_risco``), sem vazamento.

    Garantias verificadas aqui (levantam :class:`BaseInvalidaError` se falharem):

    - as colunas preditoras existem, estão na ordem oficial e são numéricas;
    - a variável-alvo existe e é binária;
    - nenhuma coluna derivada do rótulo (``risk_score`` e afins) entra em ``X``;
    - não há valores ausentes em ``X`` nem em ``y``.
    """
    validar_base(df)

    derivadas = [coluna for coluna in COLUNAS_AUXILIARES_IGNORADAS if coluna in df.columns]
    intrusas = [coluna for coluna in derivadas if coluna in COLUNAS_PREDITORAS]
    if intrusas:
        raise BaseInvalidaError(
            "Vazamento de alvo: colunas derivadas do rotulo presentes nas features: "
            + ", ".join(intrusas)
        )

    X = df.loc[:, list(COLUNAS_PREDITORAS)].copy()
    y = df.loc[:, VARIAVEL_ALVO].copy()

    if list(X.columns) != list(COLUNAS_PREDITORAS):
        raise BaseInvalidaError("A ordem/estrutura das features nao corresponde ao contrato do modelo.")
    if X.isna().any().any() or y.isna().any():
        raise BaseInvalidaError("Ha valores ausentes em X ou y.")
    return X, y


def sha256_da_base(df: pd.DataFrame) -> str:
    """Impressão digital da base usada no treino (vai para o metadata do modelo).

    O hash é calculado sobre a serialização CSV com as mesmas opções nos dois lados do
    cálculo (treino e verificação), permitindo detectar qualquer mudança na base.
    """
    return hashlib.sha256(df.to_csv(index=False).encode("utf-8")).hexdigest()


def divisao_treino_teste(
    X: pd.DataFrame,
    y: pd.Series,
    semente: int = SEMENTE,
    tamanho_teste: float = TAMANHO_TESTE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Divisão estratificada 80/20 — única e compartilhada por todos os modelos.

    Manter uma única função evita que o notebook e os módulos usem partições
    diferentes e comparações inválidas. O conjunto de teste é usado apenas para medir.
    """
    if not 0 < tamanho_teste < 1:
        raise ErroDeDados(f"'tamanho_teste' deve estar em (0, 1); recebido {tamanho_teste!r}.")
    return train_test_split(
        X, y, test_size=tamanho_teste, random_state=semente, stratify=y
    )


# --------------------------------------------------------------------------- #
# Caso de simulação do material oficial (data/novo_paciente.py)
# --------------------------------------------------------------------------- #


def normalizar_caso_de_paciente(caso: Mapping[str, Any]) -> dict[str, float]:
    """Aplica as correções documentadas do material oficial a um caso de paciente.

    O arquivo da aula usa ``"disponibilidade recursos"`` (com espaço), enquanto a base
    gerada usa ``disponibilidade_recursos``. A correção é explícita e não altera o
    arquivo original.
    """
    normalizado: dict[str, float] = {}
    for chave, valor in caso.items():
        chave_corrigida = RENOMEAR_COLUNAS_OFICIAIS.get(str(chave).strip(), str(chave).strip())
        normalizado[chave_corrigida] = float(valor)
    return normalizado


def carregar_caso_do_paciente_da_aula(raiz: str | Path | None = None) -> dict[str, float]:
    """Lê o caso de simulação de ``data/novo_paciente.py`` sem executar pandas.

    O arquivo oficial é um fragmento de célula (não importa ``pandas`` e não define
    ``pd``). Para preservá-lo intacto, executamos o código com um ``pd`` substituto que
    apenas devolve as linhas declaradas, e então normalizamos os nomes de coluna.
    """
    codigo = ler_arquivo_oficial(NOME_ARQUIVO_NOVO_PACIENTE, raiz=raiz)

    class _PandasMinimo:
        """Substituto de ``pd`` suficiente para capturar o DataFrame declarado."""

        @staticmethod
        def DataFrame(dados: Any) -> Any:
            return dados

    namespace: dict[str, Any] = {"pd": _PandasMinimo}
    try:
        exec(compile(codigo, f"{PASTA_DADOS}/{NOME_ARQUIVO_NOVO_PACIENTE}", "exec"), namespace)  # noqa: S102
    except Exception as erro:  # pragma: no cover
        raise BaseInvalidaError(
            f"Nao foi possivel interpretar '{PASTA_DADOS}/{NOME_ARQUIVO_NOVO_PACIENTE}': "
            f"{type(erro).__name__}: {erro}"
        ) from erro

    caso = namespace.get("novo_caso")
    if caso is None:
        raise BaseInvalidaError(
            f"'{PASTA_DADOS}/{NOME_ARQUIVO_NOVO_PACIENTE}' nao define a variavel 'novo_caso'."
        )
    if isinstance(caso, list):
        if not caso:
            raise BaseInvalidaError("O caso do paciente esta vazio.")
        caso = caso[0]
    elif isinstance(caso, pd.DataFrame):  # tolera execucao com pandas real
        caso = caso.iloc[0].to_dict()
    return normalizar_caso_de_paciente(caso)