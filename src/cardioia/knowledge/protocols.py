"""Base de protocolos do CardioIA (Fase 6) — camada **determinística**, sem LLM.

Fonte oficial: ``data/base_conhecimento.py`` (lido, nunca editado). Cada protocolo textual
da aula é associado, **por posição explícita e validada**, a uma regra numérica em Python.

Princípios desta camada:

- **Operadores estritos**: ``spo2_media == 88`` NÃO satisfaz ``spo2_media < 88``; o limiar
  não dispara. Nenhum limite foi "arredondado" ou reinterpretado.
- **Sem correspondência textual aproximada**: o vínculo protocolo↔regra é uma lista fixa e
  ordenada, validada contra o texto oficial por marcadores normalizados. Se o arquivo
  oficial ganhar/perder uma regra ou mudar de ordem, o carregamento falha em vez de casar
  silenciosamente com a regra errada.
- **Sem duplicatas**: cada protocolo é avaliado exatamente uma vez.
- **Somente Python decide**: o LLM nunca escolhe protocolo, nunca altera limiar e nunca
  cria regra nova. Ele recebe o resultado já calculado.

Uso típico:

    from cardioia.knowledge import protocols

    avaliacao = protocols.avaliar_protocolos(paciente, pico_risco_previsto=1)
    for aplicado in avaliacao.aplicados:
        print(aplicado.texto_oficial, "|", aplicado.condicao_satisfeita)
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..ml import dataset, predict

#: Nome do arquivo oficial que contém a lista ``protocolos``.
NOME_ARQUIVO_CONHECIMENTO = dataset.NOME_ARQUIVO_CONHECIMENTO

#: Classe positiva do modelo de ML. A última regra oficial ("risco sistêmico alto") usa a
#: **classificação do módulo de ML**, não um limiar inventado nesta camada.
CLASSE_POSITIVA = predict.CLASSE_POSITIVA

AVISO_PROTOCOLOS = (
    "Protocolos derivados da base de conhecimento da aula, aplicados de forma "
    "deterministica sobre dados SINTETICOS. Nao constituem conduta clinica validada."
)


class ErroDeProtocolos(Exception):
    """Erro base da camada de protocolos."""


class BaseDeConhecimentoInvalidaError(ErroDeProtocolos):
    """O arquivo oficial de protocolos não corresponde ao contrato esperado."""


class DadosDoPacienteInvalidosError(ErroDeProtocolos):
    """O paciente recebido não permite avaliar as condições com segurança."""


# --------------------------------------------------------------------------- #
# Estruturas
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Condicao:
    """Regra numérica determinística associada a um protocolo oficial."""

    coluna: str | None
    """Campo do paciente usado na regra. ``None`` na regra que depende do modelo de ML."""

    operador: str
    """``"<"``, ``">"`` ou ``"=="``. Operadores são estritos (o limiar não dispara)."""

    limiar: float
    """Valor de referência do material oficial (não alterar)."""

    usa_modelo: bool = False
    """``True`` somente na regra de risco sistêmico (usa a classe prevista)."""

    @property
    def descricao(self) -> str:
        """Descrição legível, ex.: ``spo2_media < 88``."""
        if self.usa_modelo:
            return f"pico_risco_previsto == {int(self.limiar)}"
        texto_limiar = int(self.limiar) if float(self.limiar).is_integer() else self.limiar
        return f"{self.coluna} {self.operador} {texto_limiar}"

    def avaliar(self, valor: float) -> bool:
        """Aplica o operador estrito ao valor observado."""
        if self.operador == "<":
            return valor < self.limiar
        if self.operador == ">":
            return valor > self.limiar
        if self.operador == "==":
            return valor == self.limiar
        raise ErroDeProtocolos(f"Operador nao suportado: {self.operador!r}.")


@dataclass(frozen=True)
class Protocolo:
    """Protocolo oficial + a regra determinística que o aciona."""

    ordem: int
    """Posição (1-based) no arquivo oficial — mantém a rastreabilidade do material."""

    identificador: str
    """Identificador estável (usado em logs, testes e no contrato de saída)."""

    texto_oficial: str
    """Texto do protocolo **exatamente** como está em ``data/base_conhecimento.py``."""

    acao: str
    """Trecho após o separador ``→`` (o que o protocolo determina)."""

    condicao: Condicao


@dataclass(frozen=True)
class ProtocoloAplicado:
    """Resultado da avaliação: protocolo + condição satisfeita + valor observado."""

    ordem: int
    identificador: str
    texto_oficial: str
    acao: str
    condicao_satisfeita: str
    operador: str
    limiar: float
    valor_observado: float
    origem_do_valor: str
    usa_modelo: bool


@dataclass(frozen=True)
class AvaliacaoDosProtocolos:
    """Saída completa da avaliação (aplicados + não aplicados + avisos)."""

    aplicados: tuple[ProtocoloAplicado, ...]
    nao_aplicados: tuple[Protocolo, ...]
    total_avaliado: int
    avisos: tuple[str, ...] = ()

    @property
    def quantidade(self) -> int:
        return len(self.aplicados)

    @property
    def acoes(self) -> list[str]:
        return [aplicado.acao for aplicado in self.aplicados]

    @property
    def identificadores(self) -> list[str]:
        return [aplicado.identificador for aplicado in self.aplicados]

    def resumo(self) -> str:
        """Resumo textual (é isto que a tool devolve ao modelo, que é texto)."""
        cabecalho = (
            f"{self.quantidade} de {self.total_avaliado} protocolos oficiais aplicaveis "
            "a este paciente."
        )
        if not self.aplicados:
            return cabecalho + " Nenhuma condicao oficial foi satisfeita."
        linhas = [cabecalho]
        for aplicado in self.aplicados:
            linhas.append(
                f"- {aplicado.texto_oficial} [condicao satisfeita: "
                f"{aplicado.condicao_satisfeita}; valor observado: "
                f"{aplicado.valor_observado:g}; origem: {aplicado.origem_do_valor}]"
            )
        return "\n".join(linhas)


# --------------------------------------------------------------------------- #
# Mapa explícito protocolo (<->) regra
# --------------------------------------------------------------------------- #

#: (identificador, marcador normalizado no texto oficial, condição)
#: A ordem desta tupla DEVE corresponder à ordem do arquivo oficial.
REGRAS_OFICIAIS: tuple[tuple[str, str, Condicao], ...] = (
    ("spo2_abaixo_de_88", "spo2 < 88", Condicao("spo2_media", "<", 88.0)),
    ("carga_do_sistema_acima_de_85", "carga do sistema > 85", Condicao("carga_sistema", ">", 85.0)),
    ("fc_acima_de_120", "fc > 120", Condicao("fc_media", ">", 120.0)),
    ("eventos_24h_acima_de_5", "eventos_24h > 5", Condicao("eventos_24h", ">", 5.0)),
    (
        "disponibilidade_abaixo_de_30",
        "disponibilidade_recursos < 30",
        Condicao("disponibilidade_recursos", "<", 30.0),
    ),
    ("tempo_de_espera_acima_de_90", "tempo_espera > 90", Condicao("tempo_espera", ">", 90.0)),
    ("temperatura_acima_de_38", "temperatura > 38", Condicao("temperatura", ">", 38.0)),
    (
        "densidade_de_eventos_acima_de_6",
        "densidade_eventos > 6",
        Condicao("densidade_eventos", ">", 6.0),
    ),
    ("pa_sistolica_abaixo_de_90", "pa sistolica < 90", Condicao("pa_sistolica", "<", 90.0)),
    (
        "risco_sistemico_alto",
        "risco sistemico alto",
        Condicao(None, "==", float(CLASSE_POSITIVA), usa_modelo=True),
    ),
)

#: Colunas do paciente exigidas pelas regras (as demais não são necessárias aqui).
COLUNAS_NECESSARIAS: tuple[str, ...] = tuple(
    dict.fromkeys(condicao.coluna for _, _, condicao in REGRAS_OFICIAIS if condicao.coluna)
)

#: Origem textual usada na justificativa de cada protocolo.
ORIGEM_DO_PACIENTE = "dados_do_paciente"
ORIGEM_DO_MODELO = f"classificacao do modelo de ML ({predict.NOME_ARQUIVO_MODELO})"


def _normalizar_texto(texto: str) -> str:
    """Minúsculas, sem acentos e com espaços colapsados (para casar marcadores)."""
    import unicodedata

    sem_acento = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")
    return " ".join(sem_acento.lower().split())


def _extrair_texto_oficial(protocolo: Any, indice: int) -> str:
    if not isinstance(protocolo, str) or not protocolo.strip():
        raise BaseDeConhecimentoInvalidaError(
            f"Protocolo na posicao {indice} nao e uma string valida: {protocolo!r}."
        )
    return protocolo.strip()


def _extrair_acao(texto_oficial: str, indice: int) -> str:
    for separador in ("→", "->"):
        if separador in texto_oficial:
            acao = texto_oficial.split(separador, 1)[1].strip()
            if acao:
                return acao
    raise BaseDeConhecimentoInvalidaError(
        f"Protocolo na posicao {indice} nao possui acao apos '→': {texto_oficial!r}."
    )


def construir_protocolos(textos_oficiais: list[str]) -> tuple[Protocolo, ...]:
    """Associa, por posição, cada protocolo oficial à sua regra determinística.

    Valida quantidade, ordem e marcador textual de **todos** os protocolos. Qualquer
    divergência levanta :class:`BaseDeConhecimentoInvalidaError` — preferimos falhar a
    avaliar uma regra errada.
    """
    if len(textos_oficiais) != len(REGRAS_OFICIAIS):
        raise BaseDeConhecimentoInvalidaError(
            f"O arquivo oficial possui {len(textos_oficiais)} protocolos; esta camada "
            f"implementa {len(REGRAS_OFICIAIS)}. Ajuste REGRAS_OFICIAIS antes de usar."
        )

    protocolos: list[Protocolo] = []
    for indice, (identificador, marcador, condicao) in enumerate(REGRAS_OFICIAIS, start=1):
        texto = _extrair_texto_oficial(textos_oficiais[indice - 1], indice)
        normalizado = _normalizar_texto(texto)
        if marcador not in normalizado:
            raise BaseDeConhecimentoInvalidaError(
                f"Protocolo {indice} nao corresponde a regra esperada '{marcador}': {texto!r}. "
                "A ordem/estrutura do arquivo oficial mudou."
            )
        protocolos.append(
            Protocolo(
                ordem=indice,
                identificador=identificador,
                texto_oficial=texto,
                acao=_extrair_acao(texto, indice),
                condicao=condicao,
            )
        )
    return tuple(protocolos)


def carregar_protocolos(raiz: str | Path | None = None) -> tuple[Protocolo, ...]:
    """Lê ``data/base_conhecimento.py`` (sem editá-lo) e devolve os protocolos estruturados."""
    codigo = dataset.ler_arquivo_oficial(NOME_ARQUIVO_CONHECIMENTO, raiz=raiz)
    espaco: dict[str, Any] = {}
    try:
        # ``exec`` em namespace isolado: o arquivo oficial define apenas ``protocolos``.
        exec(compile(codigo, f"{dataset.PASTA_DADOS}/{NOME_ARQUIVO_CONHECIMENTO}", "exec"), espaco)
    except Exception as erro:  # pragma: no cover - depende de edicao externa do arquivo
        raise BaseDeConhecimentoInvalidaError(
            f"Falha ao interpretar '{NOME_ARQUIVO_CONHECIMENTO}': {type(erro).__name__}: {erro}"
        ) from erro

    textos = espaco.get("protocolos")
    if not isinstance(textos, list):
        raise BaseDeConhecimentoInvalidaError(
            f"'{NOME_ARQUIVO_CONHECIMENTO}' nao define a lista 'protocolos'."
        )
    return construir_protocolos(textos)


# --------------------------------------------------------------------------- #
# Avaliação determinística
# --------------------------------------------------------------------------- #


def _validar_classe_prevista(pico_risco_previsto: Any) -> int:
    """Valida a classe vinda do modelo de ML (0 ou 1). Nada é convertido silenciosamente."""
    if isinstance(pico_risco_previsto, bool) or not isinstance(pico_risco_previsto, int):
        raise DadosDoPacienteInvalidosError(
            "'pico_risco_previsto' deve ser o inteiro 0 ou 1 vindo do modelo de ML; "
            f"recebido {pico_risco_previsto!r} ({type(pico_risco_previsto).__name__})."
        )
    if pico_risco_previsto not in (0, CLASSE_POSITIVA):
        raise DadosDoPacienteInvalidosError(
            f"'pico_risco_previsto' deve ser 0 ou {CLASSE_POSITIVA}; recebido {pico_risco_previsto}."
        )
    return pico_risco_previsto


def _renomear_campos_do_paciente(paciente: Any) -> dict[str, Any]:
    """Renomeia para os nomes padronizados **sem converter valores**.

    Diferente de ``dataset.normalizar_caso_de_paciente`` (que faz ``float(valor)``), aqui os
    valores permanecem **crus**: é o que permite detectar ``"85"`` (texto) ou ``True``
    (booleano) em vez de aceitá-los silenciosamente. A conversão acontece depois, já validada.
    """
    if not isinstance(paciente, Mapping):
        raise DadosDoPacienteInvalidosError(
            "Os dados do paciente devem ser um mapeamento (dict) de campo -> valor; "
            f"recebido {type(paciente).__name__}."
        )
    renomeado: dict[str, Any] = {}
    for chave, valor in paciente.items():
        nome_padronizado = dataset.RENOMEAR_COLUNAS_OFICIAIS.get(str(chave).strip(), str(chave).strip())
        renomeado[nome_padronizado] = valor
    return renomeado


def _normalizar_e_validar_paciente(paciente: Any) -> dict[str, float]:
    """Valida apenas as colunas exigidas pelas regras e devolve os valores convertidos.

    Reaproveita a validação de ``predict`` (mesma semântica: sem conversão silenciosa de
    texto, booleano, ``NaN``/``inf`` ou negativo), restrita às colunas que os protocolos
    realmente usam. Campos extras do paciente são ignorados aqui.
    """
    renomeado = _renomear_campos_do_paciente(paciente)

    ausentes = [coluna for coluna in COLUNAS_NECESSARIAS if coluna not in renomeado]
    if ausentes:
        raise DadosDoPacienteInvalidosError(
            "Campos obrigatorios ausentes para avaliar os protocolos: "
            + ", ".join(ausentes)
            + ". Os nomes esperados sao os da base gerada; a normalizacao cobre a grafia "
            "'disponibilidade recursos' (com espaco) usada no material da aula."
        )

    try:
        valores, _ = predict.validar_paciente(
            renomeado, features=list(COLUNAS_NECESSARIAS), permitir_campos_extras=True
        )
    except predict.PacienteInvalidoError as erro:
        raise DadosDoPacienteInvalidosError(str(erro)) from erro
    return valores


def avaliar_protocolos(
    paciente: Any,
    pico_risco_previsto: int,
    protocolos: tuple[Protocolo, ...] | None = None,
    raiz: str | Path | None = None,
) -> AvaliacaoDosProtocolos:
    """Avalia as regras oficiais e devolve **somente** os protocolos aplicáveis.

    Args:
        paciente: dados do paciente (``dict``; aceita a grafia do material da aula).
        pico_risco_previsto: classe prevista pelo modelo de ML (0 ou 1). É obrigatória
            porque a regra oficial ``risco_sistemico_alto`` depende da classificação do
            módulo de ML — e não de um limiar novo definido aqui.
        protocolos: catálogo já carregado (evita reler o arquivo em laços).
        raiz: raiz do projeto (para localizar ``data/base_conhecimento.py``).

    Raises:
        DadosDoPacienteInvalidosError: campos ausentes ou valores inválidos.
        BaseDeConhecimentoInvalidaError: arquivo oficial divergente do contrato.
    """
    classe = _validar_classe_prevista(pico_risco_previsto)
    valores = _normalizar_e_validar_paciente(paciente)
    catalogo = protocolos if protocolos is not None else carregar_protocolos(raiz=raiz)

    aplicados: list[ProtocoloAplicado] = []
    nao_aplicados: list[Protocolo] = []

    for protocolo in catalogo:
        condicao = protocolo.condicao
        if condicao.usa_modelo:
            valor, origem = float(classe), ORIGEM_DO_MODELO
        else:
            if condicao.coluna not in valores:  # pragma: no cover - coberto pela validacao
                raise DadosDoPacienteInvalidosError(
                    f"Coluna '{condicao.coluna}' ausente para o protocolo "
                    f"'{protocolo.identificador}'."
                )
            valor, origem = float(valores[condicao.coluna]), ORIGEM_DO_PACIENTE

        if not condicao.avaliar(valor):
            nao_aplicados.append(protocolo)
            continue

        aplicados.append(
            ProtocoloAplicado(
                ordem=protocolo.ordem,
                identificador=protocolo.identificador,
                texto_oficial=protocolo.texto_oficial,
                acao=protocolo.acao,
                condicao_satisfeita=condicao.descricao,
                operador=condicao.operador,
                limiar=float(condicao.limiar),
                valor_observado=valor,
                origem_do_valor=origem,
                usa_modelo=condicao.usa_modelo,
            )
        )

    # Garantia explícita de unicidade: se o catálogo tiver identificador repetido, falha.
    if len({aplicado.identificador for aplicado in aplicados}) != len(aplicados):
        raise ErroDeProtocolos(
            "Protocolos duplicados na resposta: o catalogo contem identificadores repetidos."
        )

    avisos: list[str] = []
    if not aplicados:
        avisos.append(
            "Nenhum protocolo oficial foi acionado: o paciente nao satisfaz nenhuma das "
            f"{len(catalogo)} condicoes da base de conhecimento da aula."
        )

    return AvaliacaoDosProtocolos(
        aplicados=tuple(aplicados),
        nao_aplicados=tuple(nao_aplicados),
        total_avaliado=len(catalogo),
        avisos=tuple(avisos),
    )


def avaliar_protocolos_seguro(
    paciente: Any,
    pico_risco_previsto: int,
    protocolos: tuple[Protocolo, ...] | None = None,
    raiz: str | Path | None = None,
) -> tuple[AvaliacaoDosProtocolos | None, list[str]]:
    """Variante tolerante a falhas, usada pela *tool* do agente.

    Em vez de propagar a exceção — o que derrubaria o ``run`` do SDK — devolve
    ``(None, [mensagem])`` para que o pipeline registre a falha **explicitamente** e não
    produza uma resposta aparentemente bem-sucedida.
    """
    try:
        avaliacao = avaliar_protocolos(
            paciente, pico_risco_previsto, protocolos=protocolos, raiz=raiz
        )
    except ErroDeProtocolos as erro:
        return None, [f"{type(erro).__name__}: {erro}"]
    return avaliacao, []