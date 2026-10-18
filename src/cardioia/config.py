"""Configuração central do CardioIA (Fase 6) — apenas constantes, sem dependências internas.

Por que existe: alguns valores estavam **duplicados** em módulos diferentes (o mesmo nome de
arquivo de artefato em ``ml/predict.py`` e ``ml/train.py``; o teto de turnos em
``agents/contexto.py``). Duplicar esse tipo de constante é uma fonte silenciosa de divergência:
renomear o artefato em um lugar quebraria o outro sem erro de importação.

Este módulo **não importa nada de ``cardioia``** (evita importação circular) e **não carrega
arquivos de configuração**:

- ``.env`` **não** é lido automaticamente. O modo real exige ``OPENAI_API_KEY`` no ambiente,
  como já acontecia antes desta etapa (nenhuma mudança de comportamento).
- nada de segredo é definido aqui: apenas os **nomes** das variáveis de ambiente.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Artefatos do modelo de ML
# --------------------------------------------------------------------------- #

PASTA_MODELOS_PADRAO = "models"
NOME_ARQUIVO_MODELO = "modelo_pico_risco.pkl"
NOME_ARQUIVO_METADADOS = "modelo_pico_risco_metadata.json"

#: Comando exato que o usuário deve rodar quando o artefato não existe (usado nas mensagens).
COMANDO_DE_TREINO = "python -m src.cardioia.ml.train"

# --------------------------------------------------------------------------- #
# Parâmetros operacionais
# --------------------------------------------------------------------------- #

#: Limiar de decisão do protótipo. ATENÇÃO: parâmetro **operacional** (0.5), escolhido para
#: coincidir com a decisão padrão de um classificador binário. **Não** é limite clínico
#: validado e não foi calibrado com dados reais.
LIMIAR_DE_DECISAO = 0.5

#: Número de árvores do Random Forest (mesmo valor do notebook e do metadata versionado).
N_ESTIMATORS_PADRAO = 300

#: Limite de turnos do ``Runner`` (proteção contra laço quando o LLM não conclui o fluxo).
MAX_TURNOS_PADRAO = 10

# --------------------------------------------------------------------------- #
# Variáveis de ambiente (somente os NOMES — os valores nunca ficam no repositório)
# --------------------------------------------------------------------------- #

VARIAVEL_DE_CHAVE = "OPENAI_API_KEY"
VARIAVEL_DE_MODELO = "OPENAI_MODEL"

__all__ = [
    "COMANDO_DE_TREINO",
    "LIMIAR_DE_DECISAO",
    "MAX_TURNOS_PADRAO",
    "NOME_ARQUIVO_METADADOS",
    "NOME_ARQUIVO_MODELO",
    "N_ESTIMATORS_PADRAO",
    "PASTA_MODELOS_PADRAO",
    "VARIAVEL_DE_CHAVE",
    "VARIAVEL_DE_MODELO",
]