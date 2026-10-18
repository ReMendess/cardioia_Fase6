"""Testes da camada de protocolos (``src/cardioia/knowledge/protocols.py``).

Cobrem: as dez regras oficiais, os **limites estritos**, seleção de múltiplos protocolos,
ausência total de protocolos, a regra de risco sistêmico (depende da classe do modelo),
normalização dos nomes de campo, ausência de duplicatas e tratamento de dados inválidos.

Nada aqui chama a API da OpenAI nem a internet.
"""

from __future__ import annotations

import pytest

from cardioia.knowledge import protocols

PROTOCOLOS_DO_PACIENTE_DA_AULA = 4


def _ids(avaliacao: protocols.AvaliacaoDosProtocolos) -> set[str]:
    return set(avaliacao.identificadores)


# --------------------------------------------------------------------------- #
# Carregamento e vínculo com o arquivo oficial
# --------------------------------------------------------------------------- #


def test_carrega_os_dez_protocolos_oficiais(raiz_do_projeto):
    catalogo = protocols.carregar_protocolos(raiz=raiz_do_projeto)
    assert len(catalogo) == 10
    assert [p.ordem for p in catalogo] == list(range(1, 11))


def test_identificadores_sao_unicos(raiz_do_projeto):
    catalogo = protocols.carregar_protocolos(raiz=raiz_do_projeto)
    identificadores = [p.identificador for p in catalogo]
    assert len(set(identificadores)) == len(identificadores)


def test_texto_oficial_e_preservado(raiz_do_projeto):
    catalogo = protocols.carregar_protocolos(raiz=raiz_do_projeto)
    assert catalogo[0].texto_oficial.startswith("Se SPO2 < 88%")
    assert catalogo[9].texto_oficial.endswith("ativar estado de emergência")
    assert catalogo[6].texto_oficial.startswith("Se temperatura > 38")


def test_cada_protocolo_tem_acao_extraida(raiz_do_projeto):
    catalogo = protocols.carregar_protocolos(raiz=raiz_do_projeto)
    for protocolo in catalogo:
        assert protocolo.acao
        assert "→" not in protocolo.acao
        assert protocolo.acao == protocolo.texto_oficial.split("→", 1)[1].strip()


def test_mapa_protocolo_regra_e_um_para_um(raiz_do_projeto):
    catalogo = protocols.carregar_protocolos(raiz=raiz_do_projeto)
    marcadores = [marcador for _, marcador, _ in protocols.REGRAS_OFICIAIS]
    assert len(marcadores) == len(catalogo)
    for protocolo, marcador in zip(catalogo, marcadores):
        normalizado = protocols._normalizar_texto(protocolo.texto_oficial)  # noqa: SLF001
        assert marcador in normalizado


def test_limiares_declarados_batem_com_o_material(raiz_do_projeto, limiares_dos_protocolos):
    catalogo = protocols.carregar_protocolos(raiz=raiz_do_projeto)
    for protocolo in catalogo:
        if protocolo.condicao.usa_modelo:
            continue
        operador, limiar = limiares_dos_protocolos[protocolo.condicao.coluna]
        assert protocolo.condicao.operador == operador
        assert protocolo.condicao.limiar == limiar


def test_regra_de_risco_sistemico_usa_a_classe_do_modelo(raiz_do_projeto):
    catalogo = protocols.carregar_protocolos(raiz=raiz_do_projeto)
    ultima = catalogo[-1]
    assert ultima.condicao.usa_modelo is True
    assert ultima.condicao.coluna is None
    assert ultima.condicao.limiar == float(protocols.CLASSE_POSITIVA)
    assert ultima.condicao.descricao == "pico_risco_previsto == 1"


def test_construir_protocolos_recusa_lista_de_tamanho_errado():
    with pytest.raises(protocols.BaseDeConhecimentoInvalidaError):
        protocols.construir_protocolos(["Se SPO2 < 88% → iniciar suporte"])


def test_construir_protocolos_recusa_ordem_trocada(raiz_do_projeto):
    catalogo = protocols.carregar_protocolos(raiz=raiz_do_projeto)
    textos = [p.texto_oficial for p in catalogo]
    textos[0], textos[1] = textos[1], textos[0]
    with pytest.raises(protocols.BaseDeConhecimentoInvalidaError):
        protocols.construir_protocolos(textos)


def test_construir_protocolos_recusa_protocolo_sem_seta():
    with pytest.raises(protocols.BaseDeConhecimentoInvalidaError):
        protocols.construir_protocolos(["Se SPO2 < 88% sem seta"] * 10)


def test_arquivo_oficial_de_conhecimento_nao_e_alterado(raiz_do_projeto):
    caminho = raiz_do_projeto / "data" / "base_conhecimento.py"
    antes = caminho.read_text(encoding="utf-8")
    protocols.carregar_protocolos(raiz=raiz_do_projeto)
    assert caminho.read_text(encoding="utf-8") == antes


def test_arquivo_ausente_gera_erro_claro(tmp_path):
    with pytest.raises(Exception) as erro:
        protocols.carregar_protocolos(raiz=tmp_path)
    assert "base_conhecimento.py" in str(erro.value)


# --------------------------------------------------------------------------- #
# Seleção de protocolos
# --------------------------------------------------------------------------- #


def test_paciente_da_aula_aciona_quatro_protocolos(raiz_do_projeto, paciente_da_aula):
    avaliacao = protocols.avaliar_protocolos(
        paciente_da_aula, pico_risco_previsto=1, raiz=raiz_do_projeto
    )
    assert avaliacao.quantidade == PROTOCOLOS_DO_PACIENTE_DA_AULA
    assert avaliacao.total_avaliado == 10
    assert _ids(avaliacao) == {
        "carga_do_sistema_acima_de_85",       # carga_sistema = 92
        "disponibilidade_abaixo_de_30",        # disponibilidade_recursos = 25
        "densidade_de_eventos_acima_de_6",     # densidade_eventos = 7
        "risco_sistemico_alto",                # classe prevista = 1
    }


def test_paciente_da_aula_nao_aciona_spo2_nem_fc(raiz_do_projeto, paciente_da_aula):
    """spo2_media=89 (< 92 do rótulo, mas > 88 do protocolo) e fc_media=118 (< 120)."""
    avaliacao = protocols.avaliar_protocolos(
        paciente_da_aula, pico_risco_previsto=1, raiz=raiz_do_projeto
    )
    assert "spo2_abaixo_de_88" not in _ids(avaliacao)
    assert "fc_acima_de_120" not in _ids(avaliacao)


def test_valores_observados_justificam_a_selecao(raiz_do_projeto, paciente_da_aula):
    avaliacao = protocols.avaliar_protocolos(
        paciente_da_aula, pico_risco_previsto=1, raiz=raiz_do_projeto
    )
    por_id = {aplicado.identificador: aplicado for aplicado in avaliacao.aplicados}
    assert por_id["carga_do_sistema_acima_de_85"].valor_observado == 92.0
    assert por_id["carga_do_sistema_acima_de_85"].limiar == 85.0
    assert por_id["carga_do_sistema_acima_de_85"].operador == ">"
    assert por_id["disponibilidade_abaixo_de_30"].valor_observado == 25.0
    assert por_id["densidade_de_eventos_acima_de_6"].valor_observado == 7.0
    assert por_id["risco_sistemico_alto"].valor_observado == 1.0
    for aplicado in avaliacao.aplicados:
        assert aplicado.condicao_satisfeita
        assert aplicado.origem_do_valor


def test_protocolos_do_material_tem_origem_do_modelo(raiz_do_projeto, paciente_da_aula):
    avaliacao = protocols.avaliar_protocolos(
        paciente_da_aula, pico_risco_previsto=1, raiz=raiz_do_projeto
    )
    do_modelo = [a for a in avaliacao.aplicados if a.usa_modelo]
    assert len(do_modelo) == 1
    assert "modelo" in do_modelo[0].origem_do_valor


def test_paciente_sem_risco_nao_aciona_protocolo(raiz_do_projeto, paciente_sem_risco):
    avaliacao = protocols.avaliar_protocolos(
        paciente_sem_risco, pico_risco_previsto=0, raiz=raiz_do_projeto
    )
    assert avaliacao.quantidade == 0
    assert avaliacao.total_avaliado == 10
    assert len(avaliacao.nao_aplicados) == 10
    assert avaliacao.avisos  # avisa explicitamente que nada foi acionado


def test_risco_sistemico_depende_da_classe(raiz_do_projeto, paciente_da_aula):
    com_risco = protocols.avaliar_protocolos(
        paciente_da_aula, pico_risco_previsto=1, raiz=raiz_do_projeto
    )
    sem_risco = protocols.avaliar_protocolos(
        paciente_da_aula, pico_risco_previsto=0, raiz=raiz_do_projeto
    )
    assert "risco_sistemico_alto" in _ids(com_risco)
    assert "risco_sistemico_alto" not in _ids(sem_risco)
    assert com_risco.quantidade == sem_risco.quantidade + 1


def test_multiplos_protocolos_simultaneos(raiz_do_projeto, paciente_sem_risco):
    critico = dict(paciente_sem_risco)
    critico.update(
        {
            "spo2_media": 85.0,        # < 88
            "carga_sistema": 90.0,     # > 85
            "fc_media": 130.0,         # > 120
            "eventos_24h": 6.0,        # > 5
            "disponibilidade_recursos": 25.0,  # < 30
            "tempo_espera": 100.0,     # > 90
            "temperatura": 39.0,       # > 38
            "densidade_eventos": 8.0,  # > 6
            "pa_sistolica": 85.0,      # < 90
        }
    )
    avaliacao = protocols.avaliar_protocolos(critico, pico_risco_previsto=1, raiz=raiz_do_projeto)
    assert avaliacao.quantidade == 10
    assert len(set(avaliacao.identificadores)) == 10  # sem duplicatas


def test_avaliacao_nao_duplica_protocolos(raiz_do_projeto, paciente_da_aula):
    avaliacao = protocols.avaliar_protocolos(
        paciente_da_aula, pico_risco_previsto=1, raiz=raiz_do_projeto
    )
    identificadores = avaliacao.identificadores
    assert len(identificadores) == len(set(identificadores))
    assert len(avaliacao.acoes) == len(set(avaliacao.acoes))


def test_ordem_dos_aplicados_segue_o_arquivo_oficial(raiz_do_projeto, paciente_da_aula):
    avaliacao = protocols.avaliar_protocolos(
        paciente_da_aula, pico_risco_previsto=1, raiz=raiz_do_projeto
    )
    ordens = [aplicado.ordem for aplicado in avaliacao.aplicados]
    assert ordens == sorted(ordens)


def test_resumo_textual_cita_condicao_e_valor(raiz_do_projeto, paciente_da_aula):
    avaliacao = protocols.avaliar_protocolos(
        paciente_da_aula, pico_risco_previsto=1, raiz=raiz_do_projeto
    )
    resumo = avaliacao.resumo()
    assert "4 de 10 protocolos oficiais aplicaveis" in resumo
    for aplicado in avaliacao.aplicados:
        assert aplicado.texto_oficial in resumo
        assert aplicado.condicao_satisfeita in resumo


# --------------------------------------------------------------------------- #
# Limites estritos (o valor exatamente no limiar NÃO dispara)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("coluna", "limiar", "identificador"),
    [
        ("spo2_media", 88.0, "spo2_abaixo_de_88"),
        ("carga_sistema", 85.0, "carga_do_sistema_acima_de_85"),
        ("fc_media", 120.0, "fc_acima_de_120"),
        ("eventos_24h", 5.0, "eventos_24h_acima_de_5"),
        ("disponibilidade_recursos", 30.0, "disponibilidade_abaixo_de_30"),
        ("tempo_espera", 90.0, "tempo_de_espera_acima_de_90"),
        ("temperatura", 38.0, "temperatura_acima_de_38"),
        ("densidade_eventos", 6.0, "densidade_de_eventos_acima_de_6"),
        ("pa_sistolica", 90.0, "pa_sistolica_abaixo_de_90"),
    ],
)
def test_valor_exatamente_no_limiar_nao_dispara(
    raiz_do_projeto, paciente_sem_risco, coluna, limiar, identificador
):
    caso = dict(paciente_sem_risco)
    caso[coluna] = limiar
    avaliacao = protocols.avaliar_protocolos(caso, pico_risco_previsto=0, raiz=raiz_do_projeto)
    assert identificador not in _ids(avaliacao)


@pytest.mark.parametrize(
    ("coluna", "valor", "identificador"),
    [
        ("spo2_media", 87.99, "spo2_abaixo_de_88"),
        ("carga_sistema", 85.01, "carga_do_sistema_acima_de_85"),
        ("fc_media", 120.01, "fc_acima_de_120"),
        ("eventos_24h", 6.0, "eventos_24h_acima_de_5"),
        ("disponibilidade_recursos", 29.99, "disponibilidade_abaixo_de_30"),
        ("tempo_espera", 90.01, "tempo_de_espera_acima_de_90"),
        ("temperatura", 38.01, "temperatura_acima_de_38"),
        ("densidade_eventos", 7.0, "densidade_de_eventos_acima_de_6"),
        ("pa_sistolica", 89.99, "pa_sistolica_abaixo_de_90"),
    ],
)
def test_valor_um_passo_alem_do_limiar_dispara(
    raiz_do_projeto, paciente_sem_risco, coluna, valor, identificador
):
    caso = dict(paciente_sem_risco)
    caso[coluna] = valor
    avaliacao = protocols.avaliar_protocolos(caso, pico_risco_previsto=0, raiz=raiz_do_projeto)
    assert identificador in _ids(avaliacao)


def test_operador_de_igualdade_nao_existe_por_acidente(raiz_do_projeto, paciente_sem_risco):
    """``spo2_media == 88`` não satisfaz ``< 88`` (operador estrito, como no material)."""
    caso = dict(paciente_sem_risco)
    caso["spo2_media"] = 88.0
    avaliacao = protocols.avaliar_protocolos(caso, pico_risco_previsto=0, raiz=raiz_do_projeto)
    assert avaliacao.quantidade == 0


# --------------------------------------------------------------------------- #
# Normalização dos campos de entrada
# --------------------------------------------------------------------------- #


def test_coluna_com_espaco_do_material_e_normalizada(raiz_do_projeto, paciente_da_aula):
    com_espaco = {
        ("disponibilidade recursos" if chave == "disponibilidade_recursos" else chave): valor
        for chave, valor in paciente_da_aula.items()
    }
    avaliacao = protocols.avaliar_protocolos(com_espaco, pico_risco_previsto=1, raiz=raiz_do_projeto)
    assert "disponibilidade_abaixo_de_30" in _ids(avaliacao)
    assert avaliacao.quantidade == PROTOCOLOS_DO_PACIENTE_DA_AULA


def test_nomes_padronizados_tambem_funcionam(raiz_do_projeto, paciente_da_aula):
    avaliacao = protocols.avaliar_protocolos(
        paciente_da_aula, pico_risco_previsto=1, raiz=raiz_do_projeto
    )
    assert avaliacao.quantidade == PROTOCOLOS_DO_PACIENTE_DA_AULA


def test_campos_extras_do_paciente_sao_ignorados(raiz_do_projeto, paciente_da_aula):
    com_extras = dict(paciente_da_aula)
    com_extras["observacao"] = "campo extra irrelevante"
    avaliacao = protocols.avaliar_protocolos(
        com_extras, pico_risco_previsto=1, raiz=raiz_do_projeto
    )
    assert avaliacao.quantidade == PROTOCOLOS_DO_PACIENTE_DA_AULA


# --------------------------------------------------------------------------- #
# Dados inválidos (falha explícita, nunca silenciosa)
# --------------------------------------------------------------------------- #


def test_campo_ausente_gera_erro(raiz_do_projeto, paciente_da_aula):
    incompleto = dict(paciente_da_aula)
    incompleto.pop("spo2_media")
    with pytest.raises(protocols.DadosDoPacienteInvalidosError) as erro:
        protocols.avaliar_protocolos(incompleto, pico_risco_previsto=1, raiz=raiz_do_projeto)
    assert "spo2_media" in str(erro.value)


@pytest.mark.parametrize("valor", ["85", True, float("nan"), float("inf"), -5.0, None])
def test_valor_invalido_gera_erro(raiz_do_projeto, paciente_sem_risco, valor):
    caso = dict(paciente_sem_risco)
    caso["carga_sistema"] = valor
    with pytest.raises(protocols.DadosDoPacienteInvalidosError):
        protocols.avaliar_protocolos(caso, pico_risco_previsto=0, raiz=raiz_do_projeto)


@pytest.mark.parametrize("classe", [True, "1", 2, -1, 0.5, None])
def test_classe_prevista_invalida_gera_erro(raiz_do_projeto, paciente_da_aula, classe):
    with pytest.raises(protocols.DadosDoPacienteInvalidosError):
        protocols.avaliar_protocolos(
            paciente_da_aula, pico_risco_previsto=classe, raiz=raiz_do_projeto
        )


def test_classe_prevista_zero_e_aceita(raiz_do_projeto, paciente_da_aula):
    avaliacao = protocols.avaliar_protocolos(
        paciente_da_aula, pico_risco_previsto=0, raiz=raiz_do_projeto
    )
    assert "risco_sistemico_alto" not in _ids(avaliacao)


def test_entrada_que_nao_e_mapping_gera_erro(raiz_do_projeto):
    with pytest.raises(protocols.DadosDoPacienteInvalidosError):
        protocols.avaliar_protocolos([1, 2, 3], pico_risco_previsto=1, raiz=raiz_do_projeto)


# --------------------------------------------------------------------------- #
# Variante tolerante a falhas (usada pela tool do agente)
# --------------------------------------------------------------------------- #


def test_avaliacao_segura_devolve_erro_em_vez_de_excecao(raiz_do_projeto, paciente_da_aula):
    incompleto = dict(paciente_da_aula)
    incompleto.pop("fc_media")
    avaliacao, erros = protocols.avaliar_protocolos_seguro(
        incompleto, pico_risco_previsto=1, raiz=raiz_do_projeto
    )
    assert avaliacao is None
    assert erros and "DadosDoPacienteInvalidosError" in erros[0]


def test_avaliacao_segura_devolve_resultado_quando_valido(raiz_do_projeto, paciente_da_aula):
    avaliacao, erros = protocols.avaliar_protocolos_seguro(
        paciente_da_aula, pico_risco_previsto=1, raiz=raiz_do_projeto
    )
    assert erros == []
    assert avaliacao is not None
    assert avaliacao.quantidade == PROTOCOLOS_DO_PACIENTE_DA_AULA
