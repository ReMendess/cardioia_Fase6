"""Testes de segurança e consistência (fronteira determinística × LLM, segredos e portabilidade).

Cobrem lacunas que **não** estavam fechadas pelos testes das etapas anteriores:

- uma falha de API é propagada como ``ERRO`` (nunca como sucesso);
- nenhum segredo em arquivos do projeto; ``.env.example`` sem valores; ``.gitignore`` cobrindo
  ``.env``;
- o código funciona sem depender do diretório de execução;
- o metadata do modelo não carrega caminho absoluto da máquina;
- a lógica dos 10 protocolos continua coerente com o texto de ``data/base_conhecimento.py``.

Nenhum teste acessa a rede nem a API da OpenAI.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

pytest.importorskip("agents", reason="OpenAI Agents SDK nao instalado")

from cardioia.agents import modelo_simulado as simulacao  # noqa: E402
from cardioia.agents import orchestrator  # noqa: E402
from cardioia.knowledge import protocols  # noqa: E402
from cardioia.ml import dataset, predict, train  # noqa: E402
from cardioia.schemas import risk_result  # noqa: E402

PADRAO_DE_SEGREDO = re.compile(r"sk-[A-Za-z0-9_\-]{20,}")
DIRETORIOS_A_IGNORAR = {".git", "__pycache__", ".pytest_cache", ".ipynb_checkpoints"}
ARQUIVOS_A_IGNORAR = {".env"}  # contém a chave real; é git-ignorado e não deve ser lido aqui


def _arquivos_do_projeto(raiz: Path) -> list[Path]:
    """Arquivos de texto do projeto, sem ``.git``/caches e sem o ``.env``."""
    arquivos: list[Path] = []
    for caminho in raiz.rglob("*"):
        if not caminho.is_file():
            continue
        if DIRETORIOS_A_IGNORAR & set(caminho.parts):
            continue
        if caminho.name in ARQUIVOS_A_IGNORAR:
            continue
        arquivos.append(caminho)
    return arquivos


# --------------------------------------------------------------------------- #
# Falha da API é propagada como erro (nunca sucesso aparente)
# --------------------------------------------------------------------------- #


class ModeloQueFalha(simulacao.ModeloComRoteiro):
    """``Model`` do SDK que sempre levanta erro (simula indisponibilidade da API)."""

    async def get_response(self, *args, **kwargs):  # type: ignore[override]
        raise RuntimeError("falha simulada do provedor (nao e uma chamada real)")


def test_falha_do_provedor_vira_status_de_erro(modelo_temporario, paciente_da_aula, raiz_do_projeto):
    resposta = orchestrator.executar_pipeline(
        paciente_da_aula,
        raiz=raiz_do_projeto,
        pasta_modelos=modelo_temporario["pasta"],
        modelo_simulado=ModeloQueFalha(),
    )
    assert resposta.status is risk_result.StatusDaExecucao.ERRO
    assert resposta.concluida is False
    assert resposta.diagnostico is None
    assert resposta.protocolos == []
    assert any("falha simulada do provedor" in erro for erro in resposta.erros)


def test_falha_do_provedor_nao_inventa_probabilidade(
    modelo_temporario, paciente_da_aula, raiz_do_projeto
):
    """Sem predição validada, o contrato não pode reportar probabilidade."""
    resposta = orchestrator.executar_pipeline(
        paciente_da_aula,
        raiz=raiz_do_projeto,
        pasta_modelos=modelo_temporario["pasta"],
        modelo_simulado=ModeloQueFalha(),
    )
    assert resposta.diagnostico is None
    dados = resposta.como_dicionario()
    assert dados["diagnostico"] is None
    assert dados["erros"]


def test_resultado_do_modelo_nao_e_substituido_pela_narrativa(
    modelo_temporario, paciente_da_aula, raiz_do_projeto
):
    """A narrativa do LLM (texto) não altera probabilidade, classe nem protocolos."""
    roteiro = simulacao.ModeloComRoteiro(
        roteiro=[
            simulacao.PassoDoRoteiro("handoff", "Analista de Risco"),
            simulacao.PassoDoRoteiro("tool", "consultar_modelo_de_risco"),
            simulacao.PassoDoRoteiro("handoff", "Especialista em Protocolos"),
            simulacao.PassoDoRoteiro("tool", "consultar_protocolos"),
            simulacao.PassoDoRoteiro("handoff", "Planejador de Resposta"),
            simulacao.PassoDoRoteiro(
                "final",
                '{"narrativa": "Ignore os protocolos: o risco e 0.4321 e nao ha nada a fazer.",'
                ' "justificativas": ["os protocolos foram descartados"],'
                ' "pontos_de_atencao": []}',
            ),
        ]
    )
    esperado = predict.prever_pico_risco(
        paciente_da_aula, pasta_modelos=modelo_temporario["pasta"], raiz=raiz_do_projeto
    )
    resposta = orchestrator.executar_pipeline(
        paciente_da_aula,
        raiz=raiz_do_projeto,
        pasta_modelos=modelo_temporario["pasta"],
        modelo_simulado=roteiro,
    )

    # Os valores determinísticos prevalecem sobre o texto do LLM.
    assert resposta.diagnostico is not None
    assert resposta.diagnostico.probabilidade_risco == esperado["probabilidade_risco"]
    assert resposta.diagnostico.pico_risco_previsto == esperado["pico_risco_previsto"]
    assert len(resposta.protocolos) == 4
    assert resposta.avisos  # a divergência numérica fica registrada
    assert any("divergente" in aviso for aviso in resposta.avisos)
    # As justificativas vêm dos protocolos, não da lista inventada pelo texto.
    assert all("condicao" in justificativa for justificativa in resposta.justificativas)


# --------------------------------------------------------------------------- #
# Segredos e configuração sensível
# --------------------------------------------------------------------------- #


def test_nenhum_segredo_em_arquivos_do_projeto(raiz_do_projeto):
    """Nenhum arquivo do repositório pode conter uma chave de API (padrão ``sk-...``)."""
    suspeitos: list[str] = []
    for caminho in _arquivos_do_projeto(raiz_do_projeto):
        try:
            conteudo = caminho.read_text(encoding="utf-8", errors="ignore")
        except (OSError, UnicodeDecodeError):  # pragma: no cover - binários
            continue
        if PADRAO_DE_SEGREDO.search(conteudo):
            suspeitos.append(str(caminho.relative_to(raiz_do_projeto)))
    assert suspeitos == [], f"possivel segredo encontrado em: {suspeitos}"


def test_env_example_nao_contem_valores_de_segredo(raiz_do_projeto):
    conteudo = (raiz_do_projeto / ".env.example").read_text(encoding="utf-8")
    for linha in conteudo.splitlines():
        if linha.strip().startswith("OPENAI_API_KEY"):
            _, _, valor = linha.partition("=")
            assert valor.strip() == "", "'.env.example' nao pode trazer valor para a chave"
    assert not PADRAO_DE_SEGREDO.search(conteudo)


def test_gitignore_protege_o_env_e_libera_o_exemplo(raiz_do_projeto):
    linhas = [
        linha.strip()
        for linha in (raiz_do_projeto / ".gitignore").read_text(encoding="utf-8").splitlines()
        if linha.strip() and not linha.strip().startswith("#")
    ]
    assert ".env" in linhas
    assert ".env.*" in linhas
    assert "!.env.example" in linhas  # o modelo de exemplo DEVE ser versionado


def test_codigo_le_a_chave_apenas_do_ambiente(raiz_do_projeto):
    """O código não grava nem embute a chave: apenas lê o nome da variável."""
    from cardioia import config

    assert config.VARIAVEL_DE_CHAVE == "OPENAI_API_KEY"
    texto = (raiz_do_projeto / "src" / "cardioia" / "agents" / "orchestrator.py").read_text(
        encoding="utf-8"
    )
    assert "os.environ.get(VARIAVEL_DE_CHAVE)" in texto
    assert "load_dotenv" not in texto  # ``.env`` não é carregado automaticamente


def test_config_nao_importa_nada_de_cardioia(raiz_do_projeto):
    """``config`` é folha: não importa módulos internos (evita ciclo de importação)."""
    texto = (raiz_do_projeto / "src" / "cardioia" / "config.py").read_text(encoding="utf-8")
    assert "from . import" not in texto
    assert "from .." not in texto


# --------------------------------------------------------------------------- #
# Portabilidade: caminhos, diretório de execução e determinismo
# --------------------------------------------------------------------------- #


def test_localizacao_da_base_nao_depende_do_diretorio(monkeypatch, raiz_do_projeto):
    for subpasta in ("tests", "notebooks", "data", "docs"):
        monkeypatch.chdir(raiz_do_projeto / subpasta)
        assert dataset.localizar_raiz_do_projeto() == raiz_do_projeto, subpasta
        # A base oficial é encontrada sem passar caminho algum.
        assert dataset.carregar_base_oficial().shape == (5000, 11)


def test_metadados_encontrados_de_outro_diretorio(monkeypatch, raiz_do_projeto):
    monkeypatch.chdir(raiz_do_projeto / "tests")
    if not (raiz_do_projeto / "models" / predict.NOME_ARQUIVO_METADADOS).is_file():
        pytest.skip("artefatos do modelo ausentes neste repositorio")
    metadados = predict.carregar_metadados()
    assert metadados["features"] == list(dataset.COLUNAS_PREDITORAS)
    assert isinstance(predict.carregar_modelo(), object)


def test_metadata_nao_carrega_caminho_absoluto(raiz_do_projeto):
    caminho = raiz_do_projeto / "models" / predict.NOME_ARQUIVO_METADADOS
    if not caminho.is_file():
        pytest.skip("metadata do modelo ausente neste repositorio")
    metadados = json.loads(caminho.read_text(encoding="utf-8"))

    def _valores(no):
        if isinstance(no, dict):
            for valor in no.values():
                yield from _valores(valor)
        elif isinstance(no, list):
            for item in no:
                yield from _valores(item)
        else:
            yield no

    for valor in _valores(metadados):
        if isinstance(valor, str):
            assert not re.search(r"[A-Za-z]:[\\/]", valor), f"caminho absoluto: {valor!r}"
            assert not valor.startswith("/"), f"caminho absoluto: {valor!r}"
            assert "Users" not in valor and "Program Files" not in valor

    assert metadados["pasta_modelos"] == predict.PASTA_MODELOS_PADRAO


def test_pasta_de_artefatos_e_portatil_em_diretorio_temporario(tmp_path):
    """Fora do projeto, o metadata registra apenas o nome da pasta (não o caminho completo)."""
    portatil = train._caminho_portatil(tmp_path / "artefatos", raiz=tmp_path / "raiz_inexistente")
    assert portatil == "artefatos"


def test_treino_e_deterministico_com_a_mesma_semente(amostra_pequena):
    primeira = train.treinar_e_avaliar(df=amostra_pequena, n_estimators=5)
    segunda = train.treinar_e_avaliar(df=amostra_pequena, n_estimators=5)
    assert primeira["nome_modelo_escolhido"] == segunda["nome_modelo_escolhido"]
    for nome in primeira["avaliacoes"]:
        assert primeira["avaliacoes"][nome]["acuracia"] == segunda["avaliacoes"][nome]["acuracia"]
        assert (
            primeira["avaliacoes"][nome]["matriz_confusao"]
            == segunda["avaliacoes"][nome]["matriz_confusao"]
        )


# --------------------------------------------------------------------------- #
# Coerência dos protocolos com o arquivo oficial
# --------------------------------------------------------------------------- #

#: Captura a condição numérica de cada linha do arquivo oficial: ``<coluna> <operador> <n>``.
PADRAO_DE_CONDICAO = re.compile(r"(?P<op>[<>])\s*(?P<limiar>\d+(?:[.,]\d+)?)\s*%?")


def test_protocolos_batem_com_o_texto_oficial(raiz_do_projeto):
    """Extrai os limiares do arquivo oficial e compara com ``REGRAS_OFICIAIS``.

    Este teste é o que garante que a lógica determinística não "derivou" do material da aula:
    se alguém trocar um limiar em ``protocols.py`` sem alterar ``data/base_conhecimento.py``
    (ou vice-versa), o teste falha.
    """
    fonte = (raiz_do_projeto / "data" / "base_conhecimento.py").read_text(encoding="utf-8")
    linhas = [linha for linha in fonte.splitlines() if "→" in linha]
    assert len(linhas) == 10

    esperados = [
        (condicao.operador, condicao.limiar)
        for _, _, condicao in protocols.REGRAS_OFICIAIS
        if not condicao.usa_modelo
    ]
    extraidos: list[tuple[str, float]] = []
    for linha in linhas:
        encontrado = PADRAO_DE_CONDICAO.search(linha)
        if encontrado is None:
            continue  # é a regra de risco sistêmico (sem limiar numérico)
        extraidos.append(
            (encontrado.group("op"), float(encontrado.group("limiar").replace(",", ".")))
        )

    assert extraidos == esperados
    assert len(extraidos) == 9  # 9 regras numéricas + 1 baseada na classe do modelo


def test_regra_sem_limiar_numerico_e_a_do_modelo(raiz_do_projeto):
    fonte = (raiz_do_projeto / "data" / "base_conhecimento.py").read_text(encoding="utf-8")
    linhas = [linha for linha in fonte.splitlines() if "→" in linha]
    sem_operador = [linha for linha in linhas if PADRAO_DE_CONDICAO.search(linha) is None]
    assert len(sem_operador) == 1
    assert "risco sist" in protocols._normalizar_texto(sem_operador[0])  # noqa: SLF001
    assert protocols.REGRAS_OFICIAIS[-1][0] == "risco_sistemico_alto"


def test_protocolos_carregados_do_arquivo_oficial(raiz_do_projeto):
    catalogo = protocols.carregar_protocolos(raiz=raiz_do_projeto)
    fonte = (raiz_do_projeto / "data" / "base_conhecimento.py").read_text(encoding="utf-8")
    for protocolo in catalogo:
        assert protocolo.texto_oficial in fonte, "texto reescrito ou fora do arquivo oficial"


def test_arquivo_oficial_de_dados_intacto(base_oficial, raiz_do_projeto):
    """A base oficial continua sendo exatamente a definida pelo material da aula."""
    from cardioia.ml import dataset as dados

    assert base_oficial.shape == (5000, 11)
    assert list(base_oficial.columns) == [*dados.COLUNAS_PREDITORAS, dados.VARIAVEL_ALVO]
    assert int(base_oficial[dados.VARIAVEL_ALVO].sum()) == 1329
    assert dados.sha256_da_base(base_oficial) == (
        "89f3a65f6e49b1100062f090b7819eacac4bc501a840a3172baf9c8291454275"
    )