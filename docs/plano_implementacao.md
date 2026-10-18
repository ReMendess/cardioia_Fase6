# CardioIA — Fase 6 · Plano de Implementação (Etapa 1)

> Documento de decisões técnicas. Registra o que **existe** no repositório (inspecionado,
> não presumido), as inconsistências encontradas, a arquitetura escolhida, as regras de
> implementação e a sequência de trabalho.
>
> **Escopo desta etapa:** inspeção + plano + notebook de Machine Learning.
> Os quatro agentes (OpenAI Agents SDK) **não** são implementados aqui.

---

## 1. Inventário real do repositório

Levantamento feito com `git ls-files --stage`, `Get-ChildItem -Recurse -Force` e leitura
integral dos arquivos de `data/`.

### 1.1 Arquivos com conteúdo

| Caminho | Bytes | Formato real | No git | Papel |
|---|---:|---|---|---|
| `README.md` | 20.835 | Markdown | sim | Descreve **a fase anterior** (AIReport Gen-Experience: RAG/FastAPI/Expo). Somente o H1 foi trocado para `# CardioIA - Fase 6`. |
| `assets/logo-fiap.png` | 28.757 | PNG | sim | Logo da FIAP. |
| `data/base_dados.py` | 1.034 | **Python (gerador)** | **não** (`?? data/`) | Base sintética: `np.random.seed(42)`, `n=5000`, 10 features + alvo derivado por regra. |
| `data/base_conhecimento.py` | 668 | **Python (lista)** | **não** | `protocolos = [ ... ]` — 10 regras em texto natural. |
| `data/novo_paciente.py` | 292 | **Python (fragmento)** | **não** | `novo_caso = pd.DataFrame([...])` — caso de simulação; **não é módulo executável**. |

### 1.2 Placeholders vazios (0 byte) já versionados

`requirements.txt`, `src/__init__.py`, `src/config.py`, `src/main.py`,
`src/agents/{__init__,orchestrator,protocol_specialist,response_planner,risk_analyst}.py`,
`src/knowledge/{__init__,protocols}.py`, `src/ml/{__init__,dataset,evaluate,predict,train}.py`,
`src/schemas/{__init__,risk_result}.py`, `tests/{test_dataset,test_prediction,test_protocols}.py`.

Esses nomes **definem a arquitetura pretendida da Fase 6** e serão reaproveitados como estão
(nenhum arquivo será renomeado ou removido).

### 1.3 Pastas vazias

`docs/`, `models/`, `notebooks/` e `src/cardioia/` (esta última é resquício, sem finalidade —
mantida intacta por não haver autorização para remoção).

### 1.4 Ausências confirmadas

- Nenhum CSV, JSON, TXT, IPYNB, PDF ou `.env` no repositório.
- Nenhum modelo salvo.
- Nenhum `.gitignore`.
- Nenhum arquivo ilegível ou corrompido: **os 3 arquivos de `data/` foram lidos integralmente**.

---

## 2. Material oficial: conteúdo literal

### 2.1 `data/base_dados.py`

- Semente global `np.random.seed(42)` e `n = 5000`.
- Colunas geradas: `idade_media` N(55,12) clip 18–90 · `carga_sistema` U(30,100) ·
  `tempo_espera` N(45,20) clip 5–180 · `fc_media` N(85,15) clip 50–150 ·
  `spo2_media` N(95,3) clip 70–100 · `pa_sistolica` N(125,20) clip 80–200 ·
  `temperatura` N(24,5) clip 10–40 · `densidade_eventos` Poisson(3) · `eventos_24h` Poisson(2) ·
  `disponibilidade_recursos` U(20,100).
- Rótulo: `risk_score` = soma de 6 indicadores booleanos e
  `df["pico_risco"] = (risk_score >= 2).astype(int)`, com os limiares do **próprio gerador**:

| Indicador | Limiar no rótulo |
|---|---|
| `carga_sistema` | `> 80` |
| `tempo_espera` | `> 60` |
| `spo2_media` | `< 92` |
| `fc_media` | `> 110` |
| `eventos_24h` | `> 4` |
| `disponibilidade_recursos` | `< 40` |

### 2.2 `data/base_conhecimento.py`

Lista `protocolos` com 10 strings (regras em linguagem natural), incluindo
`"Se risco sistêmico alto → ativar estado de emergência"`.

### 2.3 `data/novo_paciente.py`

Um único DataFrame com o caso da aula; **sem `import pandas`** e com a coluna escrita como
`"disponibilidade recursos"` (com espaço).

### 2.4 Verificações reprodutíveis executadas (Etapa 1)

Executadas em memória, sem escrever arquivos:

| Verificação | Resultado medido |
|---|---|
| Reprocessar o gerador oficial | shape `(5000, 11)`, 0 nulos, 0 duplicatas |
| Tipos | 8 features `float64`; `densidade_eventos` e `eventos_24h` `int32`; alvo `int64` |
| Distribuição do alvo | `0` → 3671 (**73,42%**); `1` → 1329 (**26,58%**) |
| Distribuição de `risk_score` | 0→1601 · 1→2070 · 2→1029 · 3→262 · 4→36 · 5→2 |
| Executar `data/novo_paciente.py` como está | `NameError: name 'pd' is not defined` |
| Caso da aula pela regra determinística | `6` indicadores ativos → `risk_score=6` → **`pico_risco=1`** |

---

## 3. Inconsistências e como cada uma é tratada

| # | Inconsistência | Tratamento |
|---|---|---|
| 1 | `novo_paciente.py` não importa `pandas` ⇒ não executa | Notebook monta o caso a partir de um `dict` explícito, documentando a correção. |
| 2 | Coluna `"disponibilidade recursos"` (com espaço) ≠ `disponibilidade_recursos` | Corrigida no notebook, com a divergência impressa como evidência. |
| 3 | Limiares do rótulo ≠ limiares dos protocolos (carga 80/85, FC 110/120, eventos 4/5, disponibilidade 40/30, SpO2 92/88, espera 60/90) | Duas camadas **separadas**: o rótulo (ML) e os protocolos (determinísticos). Tabela comparativa no notebook. |
| 4 | `temperatura` é N(24,5) (ambiente) mas há protocolo para `> 38 °C` | Documentado: o protocolo raramente dispara; a contagem real de casos é impressa. |
| 5 | `pa_sistolica < 90` tem protocolo mas **não** entra no rótulo | Documentado; nunca usado como feature derivada do alvo. |
| 6 | `densidade_eventos` não participa do rótulo | Documentado; permanece como feature preditora (pode ser ruído para o modelo). |
| 7 | Rótulo determinístico e sem ruído ⇒ teto de desempenho artificialmente alto | Declarado explicitamente no notebook e na seção 7 deste plano: **métrica alta aqui é artefato da construção sintética, não validação clínica**. |
| 8 | README descreve outro projeto (fase anterior) | Não é alterado nesta etapa; reescrita prevista para a etapa de documentação final. |
| 9 | `requirements.txt` vazio | Preenchido com versões verificadas. |
| 10 | 20 arquivos vazios em `src/` e `tests/` | Reaproveitados como estrutura alvo (sem renomear nem apagar). |
| 11 | Pasta `src/cardioia/` vazia e órfã | Mantida intacta; remoção depende de autorização. |
| 12 | Sem `.gitignore` | Criado com escopo mínimo, mantendo `data/` e `models/` versionáveis. |

---

## 4. Decisões técnicas

**D1 — Fonte única de dados (o material da aula manda).**
O notebook **executa o arquivo oficial** `data/base_dados.py` sem editá-lo. A localização é
feita por busca da raiz do repositório (pasta que contém `data/base_dados.py`), nunca por
caminho absoluto. Em Colab, a ordem de resolução é:
1. arquivo local `data/base_dados.py` (clone/upload);
2. download do raw no GitHub;
3. **réplica inline idêntica ao oficial**, com aviso explícito de failover.

**D2 — Algoritmos e critério de escolha declarado *antes* da medição.**
Treinamos **Regressão Logística** (pipeline com `StandardScaler`, interpretável) e
**Random Forest** (não linear, lida bem com limiares). Ambos no **mesmo** `train_test_split`
estratificado 80/20 (`random_state=42`). Critério pré-declarado:
**maior recall da classe `pico_risco=1`** (falso negativo é o erro mais custoso);
desempate por acurácia; depois por interpretabilidade. O notebook imprime o critério antes de
treinar e aplica exatamente esse critério.

**D3 — Ausência de vazamento.**
Features = as 10 colunas originais. `risk_score` **nunca** é criado como feature; nenhuma
variável derivada do alvo entra em `X`. Verificações com `assert` (conjunto de colunas, ausência
de `risk_score`, nulos, interseção de índices treino/teste).

**D4 — Regras numéricas são código Python, não LLM.**
A regra do gerador e os 10 protocolos são implementados deterministicamente em Python. O LLM,
na etapa seguinte, apenas interpreta/redige; ele não decide limiares.

**D5 — Artefatos com rastreabilidade.**
`models/modelo_pico_risco.pkl` (joblib) + `models/modelo_pico_risco_metadata.json` contendo
features e ordem, semente, split, critério, métricas de teste, matriz de confusão, versões de
`scikit-learn`/`numpy`/`pandas`/Python, SHA-256 da base usada e aviso de dados sintéticos.

**D6 — Escopo controlado.**
Nenhuma dependência além de `numpy`, `pandas`, `scikit-learn` e `joblib` nesta etapa
(`pydantic` e `openai-agents` ficam comentados no `requirements.txt` para a etapa dos agentes).

---

## 5. Arquitetura alvo (estrutura mínima)

```text
cardioia_Fase6/
├── README.md                      # reescrita prevista para a etapa final
├── requirements.txt               # preenchido nesta etapa
├── .gitignore                     # criado nesta etapa
├── data/                          # MATERIAL DA AULA — preservado sem edicao
│   ├── base_dados.py              # gerador oficial (fonte da verdade)
│   ├── base_conhecimento.py       # 10 protocolos em texto
│   └── novo_paciente.py           # caso de simulacao (corrigido no notebook)
├── docs/
│   ├── plano_implementacao.md     # este documento
│   ├── arquitetura.md             # diagrama (etapa dos agentes)
│   └── relatorio_tecnico.md       # etapa final
├── notebooks/
│   └── treinamento_modelo.ipynb   # ENTREGA desta etapa
├── models/
│   ├── .gitkeep
│   ├── modelo_pico_risco.pkl              # gerado pelo notebook
│   └── modelo_pico_risco_metadata.json    # gerado pelo notebook
├── src/
│   ├── ml/{dataset,train,evaluate,predict}.py   # extracao do notebook (etapa 2)
│   ├── knowledge/protocols.py                   # protocolos (etapa 3)
│   ├── schemas/risk_result.py                   # validacao Pydantic (etapa 3)
│   ├── agents/{risk_analyst,protocol_specialist,response_planner,orchestrator}.py
│   ├── config.py · main.py
└── tests/{test_dataset,test_prediction,test_protocols}.py
```

Fluxo de dados desta etapa:

```text
data/base_dados.py  ──(exec)──▶  df (5000x11)  ──▶  X (10 features) / y (pico_risco)
                                                        │
                                     train_test_split(estratificado, seed 42)
                                                        │
                        ┌───────────────────────────────┴──────────────────────────┐
                        ▼                                                          ▼
          Regressao Logistica (+Scaler)                                 Random Forest
                        └───────────────► metricas no MESMO teste ◄──────────────────┘
                                                        │
                                    criterio: maior recall da classe 1
                                                        ▼
                        models/modelo_pico_risco.pkl + *_metadata.json
                                                        ▼
                             previsao de novo caso / do paciente da aula
```

---

## 6. Sequência de implementação (cada passo testável isoladamente)

| Etapa | Entrega | Como validar |
|---|---|---|
| **1 (esta)** | `docs/plano_implementacao.md`, `notebooks/treinamento_modelo.ipynb`, `requirements.txt`, `.gitignore`, `models/.gitkeep` | Executar o notebook de ponta a ponta e conferir as métricas impressas |
| **2** | `src/ml/dataset.py`, `train.py`, `evaluate.py`, `predict.py` — extração da lógica do notebook | `tests/test_dataset.py`, `tests/test_prediction.py` (`python -m unittest discover -s tests`) |
| **3** | `src/knowledge/protocols.py`, `src/schemas/risk_result.py` (Pydantic) | `tests/test_protocols.py` |
| **4** | 4 agentes + tools/handoffs/histórico/validação de saída | Execução com `OPENAI_API_KEY` e casos de teste |
| **5** | `src/main.py`, `src/config.py`, `docs/arquitetura.md` (Mermaid), `docs/relatorio_tecnico.md`, README | Demonstração ponta a ponta |

---

## 7. Limitações declaradas (obrigatórias no relatório final)

1. Os dados são **sintéticos** e o rótulo é uma função determinística de 6 variáveis com limiares
   fixos. Um modelo com acurácia muito alta aqui **reconstrói uma regra**, não demonstra
   capacidade preditiva em cenário real.
2. **Não há valor clínico**: nada neste projeto foi validado com dados ou protocolos médicos reais.
3. O desbalanceamento é moderado (26,58% de positivos) — acurácia isolada é insuficiente;
   por isso recall/precisão da classe 1 são reportados.
4. As métricas valem **apenas** para a base gerada pela semente 42 e para as versões de
   bibliotecas registradas no metadata.
5. As regras textuais de `base_conhecimento.py` divergem dos limiares do rótulo; elas são usadas
   como camada de protocolo, **não** como verdade do rótulo.

---

## 8. Como executar

### 8.1 Google Colab

1. Abrir <https://colab.research.google.com/> → **Arquivo → Fazer upload de notebook** →
   `notebooks/treinamento_modelo.ipynb`.
2. **Menu lateral (ícone de pasta) → Upload** e enviar a pasta `data/` do repositório
   (ou rodar `!git clone https://github.com/ReMendess/cardioia_Fase6.git` — se o repositório
   ainda não tiver `data/` versionado, use o upload).
3. **Ambiente de execução → Executar tudo**.
4. A última célula empacota `models/` em `.zip` e baixa os artefatos.
5. Se nenhuma base for encontrada, o notebook usa a réplica inline do gerador oficial e
   **imprime um aviso** — as métricas continuam reprodutíveis, mas registre qual caminho foi usado.

### 8.2 Local

```bash
python -m pip install -r requirements.txt
python -m nbconvert --to notebook --execute notebooks/treinamento_modelo.ipynb \
  --output treinamento_modelo_executado.ipynb
```

---

## 9. Pendências que dependem de decisão do autor

1. **`data/` não está versionado** (`?? data/`): sem commit, o `git clone` no Colab não traz a
   base. Decidir se `data/*.py` deve ser commitado (recomendado — material oficial da aula, sem PII).
2. **Commit do `.pkl`**: versionar o binário do modelo (reconstruível via semente 42) ou apenas o
   `*_metadata.json`?
3. **`src/cardioia/`** (pasta vazia e órfã): manter ou remover? A remoção só ocorrerá com
   autorização explícita.
4. **Versionamento do `openai-agents`**: fixar a versão exata no `requirements.txt` quando a
   implementação dos agentes começar.

---

## 10. Validação executada nesta etapa (resultados medidos, não estimados)

**Comando usado** (execução real, fora do Colab, com saída descartada para a pasta temporária):

```powershell
python -m nbconvert --to notebook --execute --ExecutePreprocessor.timeout=900 `
  --output-dir "$env:TEMP\nbexec" notebooks/treinamento_modelo.ipynb
```

**Ambiente:** Python 3.13.5 · numpy 2.2.6 · pandas 2.3.2 · scikit-learn 1.8.0 · joblib 1.5.3.

**Resultado da execução:** as **19 células de código** executaram até o fim, em ordem, **sem nenhum
erro** (nenhuma célula com `output_type: error`). A base veio do arquivo oficial
(`origem: arquivo oficial data/base_dados.py`) — nenhum failover foi acionado.

### 10.1 Integridade dos dados

| Verificação | Resultado |
|---|---|
| Dimensões | `(5000, 11)` |
| Valores ausentes / duplicatas | `0` / `0` |
| Distribuição do alvo | `pico_risco=0` → 3671 (73,42%) · `pico_risco=1` → 1329 (26,58%) |
| Concordância rótulo × regra reconstruída em Python | **1,0** (assert aprovado) |
| Split estratificado | treino 4000 (26,58% positivos) · teste 1000 (26,60% positivos) |
| Sobreposição treino/teste | nenhuma (assert aprovado) |

### 10.2 Métricas no conjunto de teste (1000 amostras, 266 positivas)

| Modelo | Acurácia | Recall (classe 1) | Precisão (classe 1) | F1 (classe 1) | Matriz de confusão `[[TN, FP], [FN, TP]]` |
|---|---:|---:|---:|---:|---|
| Regressão Logística (+`StandardScaler`) | 0,8160 | 0,5526 | 0,6934 | 0,6151 | `[[669, 65], [119, 147]]` |
| **Random Forest** (300 árvores) | **0,9990** | **0,9962** | 1,0000 | 0,9981 | `[[734, 0], [1, 265]]` |
| Regra determinística do material (teto) | 1,0000 | 1,0000 | 1,0000 | 1,0000 | `[[734, 0], [0, 266]]` |

**Modelo salvo: Random Forest**, escolhido pelo critério declarado **antes** da medição
(maior recall da classe 1). O único falso negativo do Random Forest tem `risk_score = 2` — caso na
fronteira exata do corte, o que é coerente com a natureza determinística do rótulo.

**Leitura obrigatória:** a acurácia de 0,999 **não** indica capacidade preditiva real. O rótulo é uma
função determinística de 6 variáveis; o Random Forest apenas reconstrói essa regra. A Regressão
Logística, sendo linear, fica em 0,816 e mostra que a fronteira de decisão não é linearmente
separável nessas features. Nada aqui tem valor clínico.

### 10.3 Artefatos gerados e verificações independentes

| Artefato | Tamanho | Verificação |
|---|---:|---|
| `models/modelo_pico_risco.pkl` | 4.713.033 bytes | Recarregado **fora do notebook** (`joblib.load`): previsão idêntica à do modelo em memória; paciente da aula → `pico_risco=1`, probabilidade 0,9900 |
| `models/modelo_pico_risco_metadata.json` | 1.839 bytes | `sha256_da_base_usada` conferido contra a base regerada em processo separado: **confere** (`89f3a65f6e49b1100062f090b7819eacac4bc501a840a3172baf9c8291454275`) |

Simulação do paciente da aula (valores de `data/novo_paciente.py`, coluna corrigida):
`risk_score = 6` (todos os 6 indicadores ativos) → regra `= 1`, modelo `= 1` (**concordam**);
4 protocolos da base de conhecimento acionados (carga > 85%, disponibilidade < 30%,
densidade_eventos > 6 e risco sistêmico alto).

### 10.4 Sobre o notebook executado

O notebook versionado é a versão **limpa** (sem saídas), para rodar no Colab. A cópia executada com
todas as saídas foi gerada apenas para validação e mantida fora do repositório; para reproduzi-la:

```powershell
python -m nbconvert --to notebook --execute notebooks/treinamento_modelo.ipynb `
  --output notebooks/treinamento_modelo_executado.ipynb
```

---

# Etapa 2 — Camada Python modularizada (`src/cardioia/ml/`)

## 11. Objetivo e estrutura implementada

A lógica necessária para a **aplicação** saiu do notebook e virou pacote reutilizável. O notebook
continua sendo entregável obrigatório — agora **consumindo** os módulos.

```text
src/cardioia/
├── __init__.py            # docstring + versão do pacote
└── ml/
    ├── __init__.py        # API pública; carga preguiçosa (PEP 562) dos submódulos
    ├── dataset.py         # autoridade sobre data/: carga, validação, X/y, regra do rótulo
    ├── evaluate.py        # métricas, matriz de confusão, relatório serializável em JSON
    ├── train.py           # treino, seleção pelo critério, persistência + CLI
    └── predict.py         # inferência com contrato validado (interface de tool)
tests/
├── conftest.py            # sys.path + fixtures (base_oficial, amostra_pequena, modelo_temporario)
├── test_dataset.py        # 33 testes
└── test_prediction.py     # 40 testes (6 são casos parametrizados de valor inválido)
conftest.py                # raiz: insere <raiz>/src no sys.path
```

### 11.1 Justificativa do caminho `src/cardioia/ml/`

Foi o **usuário** quem especificou `src/cardioia/ml/` na Etapa 2. Isso explica a pasta órfã
`src/cardioia/` detectada na Etapa 1: ela era o destino pretendido, não um resquício.
Consequência: os placeholders vazios `src/ml/{dataset,train,evaluate,predict}.py` (versionados, 0 byte)
ficaram **superados** e **não foram removidos** (restrição: não excluir arquivos). Eles permanecem
vazios e sem uso; decisão pendente do autor (seção 16).

### 11.2 Contratos dos módulos

| Módulo | Principais funções | Garantias |
|---|---|---|
| `dataset` | `localizar_raiz_do_projeto`, `carregar_base_oficial`, `validar_base`, `calcular_risk_score`, `rotulo_pela_regra`, `indicadores_ativos`, `separar_features_e_alvo`, `sha256_da_base`, `divisao_treino_teste`, `carregar_caso_do_paciente_da_aula`, `faixas_observadas` | executa o arquivo oficial sem editá-lo; erro claro se ausente (sem réplica silenciosa); valida colunas/tipos/nulos/alvo; `risk_score` nunca em `X` |
| `evaluate` | `avaliar_modelo`, `avaliar_predicoes`, `comparar_modelos`, `ordenar_por_criterio`, `acuracia_da_regra_deterministica`, `formatar_relatorio` | métricas 100% via scikit-learn; resultado JSON-serializável; métrica indefinida → `None` + aviso (sem `0.0` enganoso) |
| `train` | `construir_modelos`, `treinar_e_avaliar`, `montar_metadados`, `salvar_artefatos`, `treinar`, `resolver_pasta_modelos`, `formatar_resumo`, `main` (CLI) | mesma semente/split para todos; teste usado **uma vez**; grava `.pkl` + JSON de metadados |
| `predict` | `carregar_modelo`, `carregar_metadados`, `features_do_modelo`, `validar_paciente`, `prever_pico_risco`, `prever_varios`, `resumo_para_agente` | reordena colunas conforme o modelo; recusa campos ausentes/desconhecidos/texto/booleano/`NaN`/`inf`/negativo; avisa (não recusa) fora da faixa; **nunca** treina silenciosamente; probabilidade **crua** |

Erros próprios (mensagens instrucionais, sem `Exception` genérica): `ErroDeDados`,
`ArquivoOficialNaoEncontradoError`, `BaseInvalidaError`, `ErroDeAvaliacao`, `ErroDeTreino`,
`ErroDePrevisao`, `ModeloNaoEncontradoError`, `PacienteInvalidoError`.

### 11.3 Decisões técnicas da Etapa 2

- **D7 — Interfaces de entrada:** aceitam `dict` **ou** `DataFrame` de 1 linha (o caso da aula é um
  `DataFrame`). O contrato é **estrito**: grafia errada (`disponibilidade recursos`) falha com dica
  explícita, em vez de "adivinhar" a intenção. `permitir_campos_extras=True` existe para quem precisa
  ignorar campos adicionais (útil quando um agente anexar metadados ao payload).
- **D8 — Limiar de decisão explícito:** `predict.LIMIAR_DE_DECISAO = 0.5`, documentado como
  **operacional**, não clínico. Com 0,5 a decisão coincide com o `predict` do scikit-learn; trocar o
  limiar muda a classificação **sem** alterar a probabilidade (testado).
- **D9 — Faixas:** valores fora da faixa **observada no treino** (gravada no metadata) geram aviso;
  fora disso, valem as faixas de projeto do gerador (`dataset.FAIXAS_ESPERADAS`). Nunca são erro —
  o modelo lida com extrapolação e recusar seria pior do que sinalizar.
- **D10 — Imports relativos** dentro do pacote: permitem usar o mesmo código por
  `python -m src.cardioia.ml.train` (raiz do repo), `import cardioia.ml.train` (via `src/` no
  `sys.path`) e pelo notebook, sem gambiarra de caminho.
- **D11 — Carga preguiçosa no `__init__`:** evita o aviso do `runpy` ao executar
  `python -m src.cardioia.ml.train` (a importação ansiosa do próprio módulo alvo é o problema).
- **D12 — Sem Pydantic nesta etapa:** a validação de entrada é feita em Python puro; o
  `src/schemas/risk_result.py` (Pydantic) entra junto com os agentes, quando o SDK exigir.

## 12. Como executar (Etapa 2)

```powershell
# 1) dependencias (pytest foi instalado em user-site, sem venv no projeto)
python -m pip install -r requirements.txt

# 2) treinar e gravar models/ (a partir da RAIZ do repositorio)
python -m src.cardioia.ml.train
python -m src.cardioia.ml.train --dry-run            # mede e NAO grava
python -m src.cardioia.ml.train --quiet              # so o comparativo
python -m src.cardioia.ml.train --pasta-modelos "$env:TEMP\modelos" --semente 42 --n-estimators 300

# 3) testes automatizados (73 testes, ~3 s)
python -m pytest tests -q
python -m pytest tests/test_prediction.py -v

# 4) uso como biblioteca (a partir da raiz)
python -c "import sys; sys.path.insert(0,'src'); from cardioia.ml import predict, dataset; print(predict.resumo_para_agente(predict.prever_pico_risco(dataset.carregar_caso_do_paciente_da_aula())))"
```

Todos os caminhos são resolvidos a partir da raiz do projeto (nenhum caminho absoluto da máquina).

## 13. Validação executada (resultados reais, medidos nesta etapa)

Ambiente: Python 3.13.5 · numpy 2.2.6 · pandas 2.3.2 · scikit-learn 1.8.0 · joblib 1.5.3 ·
pytest 9.1.1 (instalado com `pip install --user`, conforme decisão do autor).

### 13.1 Testes automatizados

| Comando | Resultado |
|---|---|
| `python -m pytest tests -q` | **73 passed in 2.41s** |
| `python -m pytest tests/test_dataset.py -q` | 33 passed |
| `python -m pytest tests/test_prediction.py -q` | 40 passed |
| `python -m pytest tests --collect-only -q` | 73 tests collected |

Nenhum teste acessa internet ou API da OpenAI. O modelo usado nos testes é treinado **na hora** em
`tmp_path` (`RandomForestClassifier(n_estimators=10)` em amostra de 400 linhas): a suíte **não**
depende do `.pkl` versionado.

**Duas falhas encontradas e corrigidas durante a validação** (registro honesto):

1. `test_coluna_com_espaco_do_material_gera_erro_com_dica` falhou: a verificação de campos ausentes
   acontecia antes da de campos desconhecidos, então a dica sobre a grafia com espaço **nunca era
   alcançada** no cenário mais provável. Corrigido em `predict.py` (`_ALIAS_DE_ENTRADA`): a mensagem
   de campo ausente agora aponta o alias.
2. `test_caso_dentro_da_faixa_nao_gera_aviso` partia de premissa errada: o modelo da fixture (400
   linhas) tem faixa de `temperatura` mais estreita (máx. 37,196) do que a faixa de projeto do
   gerador, então o paciente da aula (`temperatura = 37,5`) **gera aviso corretamente**. O teste foi
   substituído por dois: `test_caso_dentro_da_faixa_de_projeto_nao_gera_aviso` e
   `test_aviso_de_faixa_nao_altera_a_previsao`.

### 13.2 Treinamento pela linha de comando

```text
python -m src.cardioia.ml.train        →  exit code 0
  origem da base             : arquivo oficial data/base_dados.py
  amostras (treino/teste)    : 4000 / 1000
  Random Forest        acuracia=0.999      recall(1)=0.9962  precisao(1)=1.0     FN=1  FP=0
  Regressao Logistica  acuracia=0.816      recall(1)=0.5526  precisao(1)=0.6934  FN=119 FP=65
  Regra deterministica acuracia=1.0        (teto, por construcao)
  Modelo escolhido           : Random Forest
  modelo  : models/modelo_pico_risco.pkl (4.713.033 bytes)
  metadados: models/modelo_pico_risco_metadata.json (3.314 bytes)
  sha256 da base usada       : 89f3a65f6e49b1100062f090b7819eacac4bc501a840a3172baf9c8291454275
```

**Igualdade com o notebook da Etapa 1:** mesmíssimos números para os dois modelos (acurácia,
matrizes, recall, precisão, F1), mesmo SHA-256 e **mesmo tamanho de `.pkl` (4.713.033 bytes)** — isto
é, a modularização não alterou o experimento, apenas o local do código.

### 13.3 Metadados gravados

Chaves preservadas da Etapa 1 (`nome_do_modelo`, `features`, `metricas_no_teste`,
`sha256_da_base_usada`, `versoes`, `aviso`) e acrescentadas: `parametros_do_estimador`,
`limiar_de_decisao` (+ observação de que não é limite clínico),
`colunas_derivadas_do_alvo_excluidas` (`["risk_score"]`), `faixas_esperadas`, `gerado_por`,
`pasta_modelos`, `falsos_negativos`/`falsos_positivos`/`macro_f1` por modelo.

### 13.4 Previsão do paciente da aula (valores de `data/novo_paciente.py`)

Executada **em processo novo**, carregando o `.pkl` do disco:

```text
modelo carregado: RandomForestClassifier | nome no metadata: Random Forest
paciente da aula -> pico_risco_previsto=1 | probabilidade_risco=0.9900 |
                    classificacao_risco=pico_de_risco | modelo=Random Forest | limiar=0.5
risk_score da regra: 6 | pico_risco da regra: 1
modelo e regra concordam: True
sha256 da base confere com o metadata: True
```

### 13.5 Notebook reexecutado (Etapa 2, consumindo os módulos)

`python -m nbconvert --to notebook --execute` (saída em pasta temporária):

- **19/19 células de código executadas, 0 células com erro**;
- `origem da base: arquivo oficial data/base_dados.py` (failover **não** acionado);
- SHA-256 da base: `89f3a65f…54275` (idêntico);
- concordância rótulo × regra reconstruída: **1.0** (assert aprovado);
- métricas: LR 0,816 / RF 0,999 — **idênticas** às da Etapa 1;
- paciente da aula: `pico_risco=1`, probabilidade 0,99, 4 protocolos acionados, modelo e regra
  concordam;
- `nbformat.validate` aprovado; notebook versionado **sem saídas** (versão limpa).

### 13.6 Integridade do material oficial

SHA-256 dos arquivos de `data/` reconferidos ao final da etapa — **inalterados** em relação à Etapa 1:

| Arquivo | SHA-256 |
|---|---|
| `data/base_dados.py` | `BAFF2238B48865A29022398BF2F09967B6E2C5ABB379E7B3CB36C20B9D0CF9A0` |
| `data/base_conhecimento.py` | `663FC3B3E658025D9DFEAD493F21F992BC802BFC184B5609BD0648A7047DC2A0` |
| `data/novo_paciente.py` | `A419E2BB2F04DAD2393EC1C9F3282CD3FB66435460165CE4E28185EE3F2AA82C` |

---

## 14. Diferenças entre o notebook v1 (Etapa 1) e o v2 (Etapa 2)

| Aspecto | v1 (Etapa 1) | v2 (Etapa 2) |
|---|---|---|
| Origem da lógica | duplicada dentro do notebook | **módulos** `src/cardioia/ml/` |
| Bootstrap | só localizava a raiz e os 3 arquivos de `data/` | localiza a raiz, insere `src/` no `sys.path`, importa os módulos e **falha com instrução clara** se o projeto não estiver no ambiente |
| Colab | `git clone` citado como plano B | upload do `.zip` do projeto como caminho principal; clone só depois de `data/`+`src/` versionados (hoje o clone dá 404 no raw) |
| Failover da base | réplica inline usada como plano B interno | réplica inline passada **explicitamente** como `codigo_fonte=` (o módulo nunca inventa base sozinho) |
| Métricas/RF/LR/split | calculados no notebook | calculados pelos módulos — **mesmos valores** |
| Avaliação | prints montados à mão | `evaluate.formatar_relatorio` (mesmo conteúdo; cabeçalho com o nome do modelo) |
| Previsão | `joblib.load` + `predict_proba` manual | `predict.prever_pico_risco` (contrato validado, limiar explícito) |
| Saída do paciente | montado no notebook | `dataset.carregar_caso_do_paciente_da_aula` (lê o arquivo oficial e normaliza a coluna) |
| Tudo mais preservado | tabela de limiares do rótulo, divergências vs protocolos, importâncias, coeficientes, falsos negativos, oráculo determinístico, camada de protocolos (seção 13.1), seção 14 (zip) e avisos de dados sintéticos | idem |

O que **permanece no notebook**: a réplica inline do gerador (failover explícito) e a camada
determinística de protocolos da seção 13.1 — que será movida para `src/knowledge/protocols.py` na
Etapa 3.

## 15. Limitações que seguem valendo

1. Métricas altas (RF 0,999) são **artefato** de o rótulo ser uma regra determinística reconstruível.
2. **Nenhum valor clínico**: dados sintéticos, limiares de projeto, sem validação com dados reais.
3. O limiar 0,5 é operacional — não é ponto de corte clínico.
4. As faixas de aviso dependem da base de treino: um modelo treinado em amostra pequena avisa mais
   (documentado e testado).
5. As métricas valem para a semente 42 e as versões registradas no metadata.

## 16. Pendências e decisões para a próxima etapa

1. **`data/` continua untracked** (`?? data/`): sem commit, o `git clone` no Colab não traz a base —
   por isso o caminho de `.zip` no notebook. Commit de `data/*.py` é recomendado (material da aula,
   sem PII).
2. **Placeholders superados**: `src/ml/{dataset,train,evaluate,predict}.py` e `src/ml/__init__.py`
   (0 byte). Manter vazios, remover (precisa de autorização) ou transformar em re-export?
3. **`src/cardioia/__init__.py` versionado?** A pasta `src/cardioia/` é nova e ainda untracked.
4. **README.md**: ainda descreve a fase anterior (AIReport/RAG/FastAPI/Expo). Reescrever para a
   Fase 6 (com `pip install`, `pytest`, `python -m src.cardioia.ml.train` e o fluxo do notebook).
5. **Versão do `openai-agents`** a fixar no `requirements.txt` quando os agentes entrarem.
6. **Etapa 3**: `src/knowledge/protocols.py` (regras da seção 13.1, agora em módulo testável),
   `src/schemas/risk_result.py` (Pydantic validando a saída de `predict`), `tests/test_protocols.py`,
   e então os 4 agentes + orquestrador expondo `predict.prever_pico_risco` como tool.

---

# Etapa 3 — Sistema multiagente (OpenAI Agents SDK)

## 17. Versão do SDK e compatibilidade real

A versão foi escolhida por **medição das restrições no PyPI contra o ambiente**, não por
"última disponível":

| Versão | Exige `openai` | Exige `pydantic` | Situação neste ambiente |
|---|---|---|---|
| 0.24.0 / 0.22.3 | `>=3.0.0,<4` | `>=2.12.2` | ❌ exigiria `openai` 3.x |
| 0.20.0 / 0.19.4 / 0.18.3 / 0.18.2 | `>=2.45.0,<3` | `>=2.12.2` | ❌ exigiria subir `openai` |
| **0.18.1 / 0.18.0 / 0.17.8** | `>=2.36.0,<3` | `>=2.12.2` | ✅ compatível com `openai 2.38.0` |
| 0.9.3 / 0.8.4 | `>=2.19.0,<3` | `>=2.12.3` | ✅ (versões antigas) |
| 0.4.2 | `>=2.2,<3` | `>=2.10` | ✅ sem upgrade de `pydantic` (SDK bem mais antigo) |

**Escolha: `openai-agents==0.18.1`** (a mais recente compatível com o `openai` já instalado),
com a API validada na tag `v0.18.1` do repositório oficial (`function_tool`, `handoff`,
`HandoffInputData`, `RunHooks`, `Model`/`ModelProvider`, `RunResult`).

### 17.1 Efeitos colaterais da instalação (registrados, não escondidos)

`python -m pip install --user "openai-agents==0.18.1"` instalou:
`openai-agents 0.18.1`, `griffelib 2.3.2`, `mcp 1.30.0`, **`pydantic 2.14.1`** (era 2.11.10),
`pydantic-core 2.50.1`, `sse-starlette 3.5.0`, **`starlette 1.8.0`**, `pywin32`, `typing-*`.

O pip sinalizou conflitos com pacotes **de outros projetos** do user-site:

| Conflito | Ação tomada |
|---|---|
| `fastapi 0.119.0` exige `starlette<0.49.0`, mas veio `1.8.0` | **Corrigido**: fixado `starlette==0.48.0` + `sse-starlette==2.3.5` (compatíveis com o `fastapi`); `import fastapi` verificado |
| `gradio 5.49.1` exige `pydantic<2.12`, mas ficou `2.14.1` | **Não corrigível com o SDK 0.18.1** (ele exige `>=2.12.2`). `import gradio` continua funcionando, mas o `gradio` fica fora da faixa declarada. Alternativa se o gradio for necessário: usar `openai-agents==0.4.2`. |

`pip install --user` altera o user-site do Python do usuário (não há `.venv` neste projeto).
Isso ficou registrado no `requirements.txt` e nesta seção.

Nenhuma chave foi gravada em código, log ou git: `OPENAI_API_KEY` é lida apenas de variável
de ambiente e o `.env` está no `.gitignore` (com `!.env.example` liberado, sem segredos).

## 18. Arquitetura efetivamente implementada

```text
python -m src.cardioia.main  (CLI)
  │  1. valida o paciente (contrato do modelo)   2. exige models/modelo_pico_risco.pkl
  ▼
orchestrator.executar_pipeline(paciente, contexto=ContextoDaExecucao)
  │
  └─ Runner.run_sync(AgenteOrquestrador, input, context=..., hooks=TrilhaDeExecucao(),
                     run_config=RunConfig(model_provider=..., tracing_disabled=True))
        │
        ├─ 1. Orquestrador ──── handoff ───▶ Analista de Risco
        ├─ 2. Analista de Risco ── tool Python ──▶ cardioia.ml.predict.prever_pico_risco
        │                          (grava a predição NO CONTEXTO)
        ├─ 3. Analista de Risco ── handoff ─▶ Especialista em Protocolos
        ├─ 4. Especialista em Protocolos ── tool Python ──▶
        │        cardioia.knowledge.protocols.avaliar_protocolos (10 regras oficiais)
        │                          (grava os protocolos NO CONTEXTO)
        ├─ 5. Especialista em Protocolos ── handoff ─▶ Planejador de Resposta
        └─ 6. Planejador de Resposta ── output_type=ContribuicaoDoPlanejador (texto)
  │
  ▼ (Python, após o run)
  RespostaDaExecucao (contrato Pydantic) montada a partir do CONTEXTO + narrativa do LLM
```

### 18.1 Adaptação exigida pelo SDK (documentada, não simulada)

O SDK **não devolve o controle ao orquestrador** depois de um handoff: dentro de um único
`Runner.run` o fluxo é unidirecional e a saída final é do **último agente**
(documentação oficial: *"Handoffs stay within a single run"*, e `final_output` é do
`last_agent`). Portanto:

- os quatro agentes participam de **um único run**, com handoffs reais;
- os **valores autoritativos** (probabilidade, protocolos) são gravados pelas tools num
  **contexto tipado** (``RunContextWrapper.context``) — que a documentação define como objeto
  **local, nunca enviado ao LLM**;
- a montagem final do contrato acontece **em Python**, depois do run, lendo o contexto. O LLM
  contribui apenas com a **narrativa** (texto), nunca com números.

Isso atende aos requisitos de "handoffs reais" e de "resultados preservados fora das mensagens
do LLM" sem fingir um controle que o SDK não oferece.

## 19. Responsabilidade de cada agente, tools e handoffs

| # | Agente (arquivo) | Responsabilidade | Tool (Python) | Handoff de saída |
|---|---|---|---|---|
| 1 | **Analista de Risco** (`risk_analyst.py`) | consultar o modelo de ML e devolver a probabilidade **real** | `consultar_modelo_de_risco` → `predict.prever_pico_risco` | → Especialista em Protocolos |
| 2 | **Especialista em Protocolos** (`protocol_specialist.py`) | determinar os protocolos oficiais aplicáveis | `consultar_protocolos` → `protocols.avaliar_protocolos_seguro` | → Planejador de Resposta |
| 3 | **Planejador de Resposta** (`response_planner.py`) | consolidar a resposta em texto (narrativa + justificativas + atenção) | — (`output_type=ContribuicaoDoPlanejador`) | — (último do fluxo) |
| 4 | **Orquestrador** (`orchestrator.py`) | validar entrada, exigir o modelo, iniciar o run, montar/validar o contrato e registrar status/erros | — | → Analista de Risco |

### 19.1 Por que as tools não têm argumentos (decisão de segurança)

As duas tools são declaradas **sem parâmetros**: elas leem o paciente validado e a predição do
``ContextoDaExecucao``. Se o paciente fosse argumento da tool, o LLM poderia alterar valores.
Assim, o LLM decide **quando** consultar, nunca **com que números**.

### 19.2 O que o LLM nunca faz

- estimar, arredondar ou "ajustar" a probabilidade (vem da tool);
- escolher, criar ou alterar protocolo/limiar (vem da camada determinística);
- definir o status da execução (é calculado em Python a partir dos fatos observados).

### 19.3 Prova observável dos handoffs

`TrilhaDeExecucao` (subclasse de `RunHooks`) implementa `on_agent_start`, `on_handoff` e
`on_tool_start` — callbacks **async** do SDK — e grava cada evento no contexto. Do run real do
paciente da aula:

```text
HANDOFFS REALIZADOS
  -> Orquestrador -> Analista de Risco
  -> Analista de Risco -> Especialista em Protocolos
  -> Especialista em Protocolos -> Planejador de Resposta

TRILHA DE EXECUCAO (12 passos)
  1. [validacao/ok] Orquestrador: entrada do paciente validada contra o contrato do modelo
  2. [agente/ok] Orquestrador ... 3. [handoff/ok] Orquestrador -> Analista de Risco
  5. [tool/ok] Analista de Risco: tool 'consultar_modelo_de_risco' chamada
  6. [tool/ok] Analista de Risco: predicao calculada por cardioia.ml.predict
  7. [handoff/ok] Analista de Risco -> Especialista em Protocolos
  9. [tool/ok] Especialista em Protocolos: tool 'consultar_protocolos' chamada
 10. [tool/ok] Especialista em Protocolos: protocolos avaliados (4 aplicaveis)
 11. [handoff/ok] Especialista em Protocolos -> Planejador de Resposta
 12. [agente/ok] Planejador de Resposta: iniciou o turno
```

## 20. Compartilhamento de resultados (contexto tipado)

`ContextoDaExecucao` (`agents/contexto.py`) é um `dataclass` **local** passado em
`Runner.run_sync(..., context=...)`. A documentação do SDK é explícita: o contexto **não é
enviado ao LLM**. Ele carrega:

| Campo | Papel |
|---|---|
| `paciente` | dados já validados (nomes padronizados) |
| `raiz`, `pasta_modelos`, `limiar` | dependências para as tools (sem caminho absoluto fixo) |
| `predicao` | **resultado cru** de `predict.prever_pico_risco` (gravado pela tool) |
| `avaliacao_de_protocolos` | resultado da camada determinística (gravado pela tool) |
| `passos`, `erros`, `handoffs` | trilha auditável e falhas explícitas |

O LLM recebe apenas o **resumo textual** que as tools devolvem (o `ToolCallOutputItem` fica no
histórico). Se a narrativa citar um número diferente do calculado, o pipeline emite aviso e
muda o status para `CONCLUIDO_COM_RESSALVAS` — o valor da tool continua sendo o oficial.

### 20.1 Histórico de mensagens

O SDK carrega o histórico entre os turnos automaticamente: no 3º turno já existem
`function_call` e `function_call_output` no `input`, e no turno final há duas chamadas de tool
com seus resultados. Isso é verificado em
`test_orchestrator.py::test_historico_do_sdk_carrega_chamadas_e_resultados` (que inspeciona os
tipos de item recebidos em cada turno). Não há persistência entre execuções: cada
`executar_pipeline` cria um contexto novo e um `RunConfig` sem `session`/`conversation_id` —
**nada é reutilizado entre runs**.

## 21. Contratos de entrada e saída (Pydantic)

`src/cardioia/schemas/risk_result.py` é o **contrato único** usado pelo orquestrador, pela CLI
e pelos testes (sem estruturas duplicadas). `extra="forbid"` em todos os modelos.

| Modelo | Campos principais | Restrições |
|---|---|---|
| `IndicadorDeterministico` | `nome`, `coluna`, `operador`, `limiar`, `valor_observado`, `satisfeito`, `origem` | `operador ∈ {<, >, ==}` |
| `ProtocoloAplicavel` | `ordem`, `identificador`, `texto_oficial`, `acao`, `condicao_satisfeita`, `operador`, `limiar`, `valor_observado`, `origem_do_valor`, `usa_modelo` | `ordem ∈ [1, 10]`; textos não vazios |
| `DiagnosticoDaPredicao` | `probabilidade_risco`, `pico_risco_previsto`, `classificacao_risco`, `modelo_utilizado`, `classe_do_estimador`, `limiar_decisao`, `regra_de_decisao`, `features_utilizadas`, `origem_do_modelo`, `avisos` | probabilidade em `[0, 1]` **sem arredondar**; classe `Literal[0, 1]` **rejeitando `bool`**; classificação coerente com a classe |
| `PassoDaExecucao` | `ordem`, `tipo`, `agente`, `detalhe`, `sucesso` | `tipo ∈ {handoff, tool, agente, validacao, erro}` |
| `RespostaDaExecucao` | `id_execucao`, `gerado_em` (UTC), `status`, `modo`, `diagnostico`, `indicadores`, `protocolos`, `justificativas`, `narrativa`, `avisos`, `limitacoes`, `erros`, `passos`, `handoffs_realizados`, `agente_final`, `versao_do_sdk`, `duracao_segundos` | ver §22 |

Detalhe importante: o pydantic, em modo não estrito, converteria `True` em `1` numa
`Literal[0, 1]` — o que faria um booleano passar como "pico de risco". Um `field_validator`
rejeita `bool` e tipos não inteiros, em coerência com `predict.validar_paciente` e
`protocols._validar_classe_prevista`.

## 22. Tratamento de erros e status (falha nunca vira sucesso)

| Status | Quando |
|---|---|
| `CONCLUIDO` | predição validada + os 3 handoffs + sem erros + nenhum aviso crítico |
| `CONCLUIDO_COM_RESSALVAS` | houve erro/aviso não fatal (ex.: narrativa divergente da probabilidade) |
| `INCOMPLETO` | falta predição validada ou o run não terminou no Planejador (handoff da cadeia não ocorreu) |
| `ERRO` | modelo ausente, entrada inválida, chave ausente no modo real, exceção do SDK (`MaxTurnsExceeded`), falha inesperada |

Regras do contrato que garantem isso:

- `status=CONCLUIDO` **exige** `diagnostico` e **proíbe** `erros` preenchidos;
- `status ∈ {ERRO, INCOMPLETO}` **exige** ao menos um item em `erros`;
- se o run termina sem predição, o orquestrador rebaixa o status e registra o motivo.

Outras proteções implementadas:

| Situação | Comportamento |
|---|---|
| `models/*.pkl` ausente | `ERRO` **antes** de qualquer chamada de LLM, com o comando de treino na mensagem; **nunca** treina |
| tool do modelo falha | mensagem `FALHA: ...` na conversa (o run não quebra) + erro no contexto |
| tool de protocolos sem predição | `FALHA` explícita (a regra de risco sistêmico depende da classe) |
| dados de protocolo inválidos | `avaliar_protocolos_seguro` devolve erro em vez de exceção |
| handoff inconsistente no roteiro simulado | registrado em `erros` como `[roteiro simulado] ...` |
| laço de turnos | `max_turns` (padrão 10) → `MaxTurnsExceeded` capturado → `ERRO` |
| modo real sem `OPENAI_API_KEY` | `ERRO` com instrução (`--modo simulado` como alternativa) |
| tracing | `tracing_disabled=True` (nenhum dado do paciente sai para a plataforma) |

## 23. Como executar o pipeline

```powershell
# 1) dependencias (ja instaladas em user-site neste ambiente)
python -m pip install -r requirements.txt

# 2) modelo de ML (se ainda nao existir)
python -m src.cardioia.ml.train

# 3) pipeline — modo SIMULADO (padrao, NAO usa a API da OpenAI)
python -m src.cardioia.main                              # paciente de data/novo_paciente.py
python -m src.cardioia.main --json                       # contrato completo em JSON
python -m src.cardioia.main --paciente caso.json         # paciente em arquivo
python -m src.cardioia.main --paciente caso.json --quiet # apenas o status

# 4) pipeline — modo REAL (exige a chave no ambiente)
$env:OPENAI_API_KEY = "sk-..."
python -m src.cardioia.main --modo real --modelo gpt-4o-mini

# 5) testes
python -m pytest tests -q                                          # 250 testes (2 pulados)
python -m pytest tests/test_integracao_real.py -v -m integracao    # so com chave
```

Códigos de saída da CLI: `0` quando o pipeline conclui (`CONCLUIDO`/`CONCLUIDO_COM_RESSALVAS`)
e `1` em `ERRO`/`INCOMPLETO` (com os motivos impressos em `stderr`).

Saída real do modo simulado com o paciente da aula:

```text
status         : CONCLUIDO
modo           : simulado
agente final   : Planejador de Resposta

PREDICAO (calculada pelo modelo de ML, nunca pelo LLM)
  modelo                 : Random Forest (RandomForestClassifier)
  probabilidade de pico  : 0.9900
  pico_risco_previsto    : 1
  classificacao de risco : pico_de_risco
  limiar de decisao      : 0.5 (probabilidade_risco >= limiar (0.5))

PROTOCOLOS APLICAVEIS (4)
  - Se carga do sistema > 85% -> abrir unidades temporarias      [valor observado: 92]
  - Se disponibilidade_recursos < 30% -> redistribuir insumos    [valor observado: 25]
  - Se densidade_eventos > 6 -> ativar plano de contingencia     [valor observado: 7]
  - Se risco sistemico alto -> ativar estado de emergencia       [classe prevista = 1]
```

## 24. Testes executados (medidos nesta etapa)

| Comando | Resultado |
|---|---|
| `python -m pytest tests -q` | **250 passed, 2 skipped in 7.42s** |
| `python -m pytest tests/test_protocols.py -q` | 61 passed |
| `python -m pytest tests/test_schemas.py -q` | 61 passed |
| `python -m pytest tests/test_agents.py -q` | 26 passed |
| `python -m pytest tests/test_orchestrator.py -q` | 29 passed |
| Etapa 2 (inalterada) | 73 passed dentro do total |
| `test_integracao_real.py` | **2 skipped** (sem `OPENAI_API_KEY`) |

Cobertura dos novos testes: as 10 regras oficiais, limites **estritos** (valor no limiar não
dispara; um passo além dispara), múltiplos protocolos, nenhum protocolo, regra de risco
sistêmico ligada à classe do modelo, normalização do nome com espaço, ausência de duplicatas,
dados inválidos; contratos (probabilidade válida/inválida/`NaN`, classes, campos obrigatórios,
campos desconhecidos, coerência status↔conteúdo, JSON-safe); tools chamando as implementações
reais; handoffs configurados; fluxo completo com os 3 handoffs; histórico de mensagens; falhas
(modelo ausente, entrada inválida, tool sem predição, handoff incompleto, roteiro inconsistente,
`MaxTurnsExceeded`, modo real sem chave, narrativa divergente).

### 24.1 Validações de ambiente executadas

| Verificação | Resultado |
|---|---|
| `import agents` resolve para o SDK (não para `src/agents/`) | ✅ `test_sdk_agents_nao_e_sombreado_pelo_repositorio` |
| `python -m src.cardioia.main` (paciente da aula) | ✅ `status=CONCLUIDO`, exit 0 |
| `python -m src.cardioia.main --json` | ✅ JSON válido: `CONCLUIDO`, probabilidade `0.99`, 4 protocolos, 3 handoffs, 12 passos |
| `python -m src.cardioia.main --quiet --paciente <arquivo>.json` | ✅ `status=CONCLUIDO`, exit 0 (paciente sem risco → 0 protocolos) |
| `python -m src.cardioia.main --modo real` sem chave | ✅ `status=ERRO`, exit 1, mensagem em `stderr` |
| Notebook reexecutado (nbconvert) | ✅ 0 erros; métricas idênticas (LR 0,816 / RF 0,999); paciente → 1 |
| `git diff --name-only -- data src/cardioia/ml notebooks models` | ✅ **vazio** (nada alterado) |
| SHA-256 de `data/*.py` | ✅ idênticos aos da Etapa 1 |
| `import fastapi` após a instalação do SDK | ✅ restaurado com `starlette==0.48.0` |

### 24.2 Tentativa de integração real com a API da OpenAI (executada)

O ambiente **tem** um `.env` com `OPENAI_API_KEY` (164 caracteres, corretamente ignorado pelo
git). Carreguei a chave apenas na sessão do comando (nunca impressa, nunca gravada em arquivo)
e executei `pytest tests/test_integracao_real.py -m integracao`:

| Cenário | Resultado |
|---|---|
| Sem `OPENAI_API_KEY` | 250 passed, **3 skipped** (motivo: chave ausente) |
| Com a chave do `.env` | **251 passed, 2 skipped** |

O que a execução real provou e o que ela **não** provou:

- ✅ a chave **autentica** e a requisição **chega** à API pelo caminho real do SDK
  (`openai_responses`): o log do SDK mostrou a resposta HTTP do provedor;
- ✅ o pipeline **não finge sucesso** quando a API falha: o erro virou
  `status=ERRO` com a mensagem original em `erros`, `diagnostico=null` e `concluida=False`
  (validado pelo teste `test_modo_real_sem_creditos_reporta_erro_e_nao_sucesso`, que **passa**
  nesse cenário);
- ❌ **não foi possível validar um run real bem-sucedido**: a conta responde
  `HTTP 429 / insufficient_quota` (*"You have no credits remaining"*). Portanto a condução do
  fluxo pelo LLM real (escolher os handoffs e chamar as tools) **não está verificada**. Quando
  houver crédito, basta reexecutar o comando — o teste valida os 3 handoffs, a probabilidade
  idêntica à da tool e o contrato final.

### 24.3 Bug real encontrado **pelo teste de integração** (e corrigido)

A primeira execução em modo real falhou com
`AttributeError: 'NoneType' object has no attribute 'get_model'`. Causa: em
`RunConfig(model=modelo, model_provider=None, ...)` eu passava `model_provider=None`
explicitamente, **sobrescrevendo o default factory** do SDK (que é um `MultiProvider`). Correção
em `orchestrator._executar_run`: os campos só entram no `RunConfig` quando fazem sentido
(`model_provider` apenas no modo simulado; `model` apenas quando informado).

Este é o valor concreto de ter escrito o teste de integração mesmo sem crédito: o defeito só
aparecia no caminho real — a suíte simulada passava por construção.

## 25. Limitações conhecidas (declaradas, não contornadas)

1. **A integração com a API real da OpenAI não foi validada em um run bem-sucedido**: a chave
   presente no `.env` autentica, mas a conta responde `HTTP 429 / insufficient_quota`
   (*"You have no credits remaining"*). O caminho real foi exercitado até a resposta do provedor
   e o erro foi tratado corretamente (`ERRO`, sem sucesso falso — §24.2). O fluxo completo com o
   LLM conduzindo os handoffs permanece **não verificado**; `tests/test_integracao_real.py`
   está pronto para validar isso quando a conta tiver crédito.
2. O modo simulado usa uma **narrativa fixa** (sem números) para exercitar a saída estruturada;
   ela não representa a qualidade de redação de um LLM real.
3. Conflito residual de ambiente: `gradio 5.49.1` declara `pydantic<2.12` e o SDK 0.18.1 exige
   `>=2.12.2`. `import gradio` funciona, mas o pacote fica fora da faixa declarada (§17.1).
4. Protocolos com limites de **projeto** (material da aula); nenhuma conduta clínica validada;
   dados **sintéticos** — nada aqui é diagnóstico ou prescrição.
5. Sem persistência entre execuções: cada run cria contexto e histórico próprios.
6. `tracing` desativado por padrão: não há observabilidade remota (dashboard) do SDK.

## 26. Pendências para a próxima etapa

1. Adicionar créditos à conta da OpenAI (ou usar outra chave) e rodar
   `python -m pytest tests/test_integracao_real.py -m integracao` para validar o run real
   completo (handoffs decididos pelo LLM).
2. Reescrever o `README.md` (ainda descreve outra fase): comandos de ML, pipeline multiagente,
   arquitetura, testes e roteiro de demonstração.
3. Produzir `docs/arquitetura.md` (diagrama Mermaid) e `docs/relatorio_tecnico.md` da entrega.
4. Decidir sobre os placeholders vazios `src/{ml,agents,knowledge,schemas}/*.py` (superados por
   `src/cardioia/**`): manter, remover (com autorização) ou converter em re-export.
5. Centralizar configurações em `src/cardioia/config.py` (hoje: defaults nos módulos + flags CLI).
6. Avaliar `output_guardrails` do SDK para validar a narrativa também no modo real — hoje a
   checagem de números divergentes é programática (Python), o que já é o mais confiável.

---

# Etapa 4 — Documentação da entrega, configuração central e testes de lacuna

## 27. Escopo e decisões da Etapa 4

| # | Decisão | Situação |
|---|---|---|
| D1 | Reescrever o `README.md` para a Fase 6 (o anterior descrevia AIReport/RAG/FastAPI/Expo e citava arquivos inexistentes) | ✅ feito |
| D2 | Criar `docs/arquitetura.md` com diagramas Mermaid | ✅ feito (5 diagramas) |
| D3 | Criar `docs/relatorio_tecnico.md` | ✅ feito |
| D4 | Gerar dois PDFs da entrega | ✅ feito (relatório 2 páginas; arquitetura 3 páginas) |
| D5 | Centralizar configurações em `src/cardioia/config.py` | ✅ feito |
| D6 | **Não** carregar `.env` automaticamente | ✅ mantido (nenhuma mudança de comportamento) |
| D7 | **Não** adicionar `output_guardrails` | ✅ mantido: a checagem de números divergentes já é programática |
| D8 | Manter os placeholders vazios e documentar `src/cardioia/**` como implementação oficial | ✅ feito (§33) |
| D9 | Corrigir o caminho absoluto gravado no metadata do modelo | ✅ feito (§29) |
| D10 | Manter `gradio` no ambiente e documentar o conflito | ✅ feito (§32) |
| D11 | Reordenar o plano e acrescentar a Etapa 4 | ✅ feito (seções 14–16 movidas para antes da Etapa 3) |

Autor e RM foram mantidos no README, com uma seção **Vídeo** contendo o marcador
`(a fornecer)` — não existe vídeo desta fase.

## 28. Configuração centralizada (`src/cardioia/config.py`)

**Problema real corrigido:** o mesmo nome de artefato estava declarado em **dois** módulos
(`ml/predict.py` e `ml/train.py`, com `train.py` importando `LIMIAR_DE_DECISAO` de `predict.py`), e
o teto de turnos vivia em `agents/contexto.py`. Renomear um artefato em um lugar quebraria o outro
sem erro de importação.

Agora existe **uma fonte única**, sem qualquer importação interna (evita ciclo de importação):

| Constante | Valor | Consumidores |
|---|---|---|
| `PASTA_MODELOS_PADRAO` | `models` | `predict`, `train`, CLI |
| `NOME_ARQUIVO_MODELO` | `modelo_pico_risco.pkl` | `predict`, `train`, CLI |
| `NOME_ARQUIVO_METADADOS` | `modelo_pico_risco_metadata.json` | `predict`, `train` |
| `COMANDO_DE_TREINO` | `python -m src.cardioia.ml.train` | mensagens de erro |
| `LIMIAR_DE_DECISAO` | `0.5` | `predict`, `train`, CLI |
| `N_ESTIMATORS_PADRAO` | `300` | `train` |
| `MAX_TURNOS_PADRAO` | `10` | `orchestrator` |
| `VARIAVEL_DE_CHAVE` / `VARIAVEL_DE_MODELO` | `OPENAI_API_KEY` / `OPENAI_MODEL` | `orchestrator`, CLI |

`predict`, `train` e `contexto` **reexportam** os nomes (`__all__`), preservando a interface
pública já usada pelo notebook, pelos testes e pela CLI. Verificação executada:

```text
predict : modelo_pico_risco.pkl 0.5 python -m src.cardioia.ml.train models
train   : modelo_pico_risco.pkl modelo_pico_risco_metadata.json 300 0.5
mesma origem (identidade do objeto): True True
```

Confirmado que o `.env` **não** é carregado automaticamente (nenhum `load_dotenv` no código) — o
modo real continua exigindo a variável no ambiente.

## 29. Correção do caminho absoluto no metadata do modelo

**Defeito encontrado:** `models/modelo_pico_risco_metadata.json` trazia
`"pasta_modelos": "C:\\Users\\Pichau\\OneDrive\\Área de Trabalho\\cardioia_Fase6\\models"` — caminho
da máquina do autor dentro de um arquivo versionado.

**Correção:** `train._caminho_portatil()` grava o caminho **relativo à raiz** (`models`); se a pasta
estiver fora do projeto (temporários dos testes), grava apenas o nome da pasta.

**Verificação de que nada mais mudou (medida, não estimada):**

| Item | Antes | Depois |
|---|---|---|
| `models/modelo_pico_risco.pkl` (SHA-256) | `F09DCF5FC25AA0AE77473D42A9F52B01E1566675F471B8F4E32952CFEBC40021` | **idêntico** |
| Chaves do metadata alteradas | — | **apenas** `pasta_modelos` |
| `metricas_no_teste`, `features`, `semente`, `sha256_da_base_usada` | — | **idênticos** |
| Caminho absoluto no JSON | presente | **ausente** |

O `git status` confirma: `modelo_pico_risco.pkl` **não** aparece como modificado (só o metadata).
Um teste novo guarda isso de forma permanente
(`test_seguranca.test_metadata_nao_carrega_caminho_absoluto`).

## 30. Documentação da entrega

| Artefato | Conteúdo | Ferramenta de geração |
|---|---|---|
| `README.md` | Apresentação, escopo acadêmico, arquitetura, instalação, execução (simulado/real/JSON), testes, resultados, limitações | Markdown escrito à mão |
| `docs/arquitetura.md` | 5 diagramas Mermaid (componentes, sequência, fronteira de confiança, dados, falhas) + tabelas | Mermaid escrito à mão |
| `docs/relatorio_tecnico.md` | Relatório em 9 seções (problema, dados, metodologia, resultados, arquitetura, protocolos, limitações, conclusão, referências) | Markdown escrito à mão |
| `docs/relatorio_tecnico.pdf` | **2 páginas**, 202.145 bytes | Markdown → HTML (Python-Markdown 3.9) → PDF (Edge headless) → verificação (PyMuPDF) |
| `docs/arquitetura_multiagente.pdf` | **3 páginas**, 510.986 bytes, diagramas vetoriais (SVG) | Mermaid CLI (`mmdc` 12.0.0 com o Edge local) → HTML → PDF (Edge headless) → verificação (PyMuPDF) |

Ferramentas efetivamente usadas: `Python-Markdown`, `PyMuPDF` 1.27.2.3, `msedge`
(`C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe`), Node/npm e
`@mermaid-js/mermaid-cli` 12.0.0 — instalado em `%TEMP%` (**fora** do repositório) com
`PUPPETEER_SKIP_DOWNLOAD=true`, reaproveitando o Edge já instalado. Não disponíveis e portanto **não**
usados: `reportlab`, `weasyprint`, `pandoc`, `playwright`.

> Detalhe técnico registrado: os dois diagramas verticais (`flowchart TD`) ficaram ilegíveis em A4
> quando ajustados por largura. Para o PDF eles são re-renderizados como `flowchart LR` (largos e
> curtos); o `.md` mantém a direção original, adequada à leitura com zoom no GitHub.

## 31. Novos testes: CLI e segurança

| Arquivo | Testes | Lacuna que fecha |
|---|---|---|
| `tests/test_cli.py` | **14** | comando real `python -m src.cardioia.main` e **código de saída**; `--json` válido **e** revalidado pelo contrato Pydantic; paciente por arquivo; JSON inválido; modelo ausente; modo real sem chave; `--help` com as flags reais; independência do diretório de execução; a CLI não cria arquivos indevidos na raiz |
| `tests/test_seguranca.py` | **17** | falha do provedor vira `status=ERRO` sem predição; narrativa do LLM **não** sobrescreve números autoritativos; nenhum segredo (`sk-...`) em arquivos do projeto; `.env.example` sem valor; `.gitignore` protegendo `.env` e liberando o exemplo; `config` sem importações internas; base oficial encontrada de qualquer diretório; metadata sem caminho absoluto; determinismo do treino; **limiares dos protocolos conferidos contra o texto do arquivo oficial**; integridade da base (dimensões, positivos e SHA-256) |

Para manter a suíte rápida, os testes de CLI usam `main(argv)` com `capsys` e apenas **um** caso abre
processo novo (para conferir o código de saída do sistema) — reduzindo o tempo do arquivo de 86 s
para ~18 s.

**Contagem medida da suíte completa:**

| Momento | Resultado |
|---|---|
| Antes da Etapa 4 | 250 passed, 3 skipped (253 coletados) |
| Depois da Etapa 4 | **281 passed, 3 skipped** (284 coletados, ~21 s) |

Com `OPENAI_API_KEY` no ambiente, os 3 testes de `test_integracao_real.py` deixam de ser pulados e
tentam a API — que responde `429 / insufficient_quota` (§34.3). Isso não é falha dos testes unitários.

## 32. Conflito de dependências: `openai-agents` × `gradio` (documentado, não resolvido)

| Pacote | Restrição declarada | Situação |
|---|---|---|
| `openai-agents 0.18.1` | `pydantic>=2.12.2` | ✅ satisfeita (`pydantic 2.14.1`) |
| `gradio 5.49.1` | `pydantic<2.12` | ❌ **incompatível** com o SDK fixado |

As duas restrições são **mutuamente insatisfazíveis** no mesmo ambiente. `import gradio` continua
funcionando, mas o pacote fica fora da faixa que ele próprio declara. Alternativas verificadas:

1. usar `openai-agents==0.4.2` (exige `pydantic>=2.10` e `openai>=2.2,<3`) — SDK bem mais antigo;
2. manter ambientes separados (um `venv` por projeto) — recomendado para uso continuado.

O conflito está registrado no `requirements.txt` e no README (§14.4). **Não** se declara resolvido.

## 33. Placeholders vazios: decisão de manter (D8)

Os **17 arquivos de 0 byte** (`src/{__init__,config,main}.py`, `src/ml/*.py`, `src/agents/*.py`,
`src/knowledge/*.py`, `src/schemas/*.py`) são **git-tracked** e fazem parte da estrutura inicial da
atividade. Verificações feitas nesta etapa:

- **nenhuma** referência externa a eles (busca em `src/cardioia`, `tests`, `notebooks`, `docs`);
- os arquivos `src/cardioia/config.py` e `src/cardioia/main.py` são **novos** (a pasta `src/cardioia/`
  é separada de `src/config.py` e `src/main.py`), portanto não há colisão de importação;
- a implementação oficial é **`src/cardioia/**`**, documentada no README e no `docs/arquitetura.md`.

Recomendação registrada (não executada, por exigir autorização): remover os placeholders em uma etapa
futura ou convertê-los em reexportações de `src/cardioia/**`.

## 34. Validação final executada (resultados medidos)

### 34.1 Suíte automatizada

```text
python -m pytest tests -q
281 passed, 3 skipped in 21.26s
```

### 34.2 Comandos do produto

| Comando | Resultado medido |
|---|---|
| `python -m src.cardioia.ml.train --dry-run` | métricas idênticas (LR 0,816 / RF 0,999) e `Artefatos: NAO gravados (modo --dry-run)` |
| `python -m src.cardioia.main --quiet` | `status=CONCLUIDO`, código de saída **0** |
| `python -m src.cardioia.main --json` | JSON válido, revalidado pelo contrato Pydantic; probabilidade crua; 3 handoffs; 4 protocolos |
| `python -m src.cardioia.main --help` | lista as 9 flags reais e o aviso de dados sintéticos |

### 34.3 Estado honesto do modo real

Continua **não validado** o run completo com LLM real: a chave autentica, a requisição chega à API e
a conta responde `HTTP 429 / insufficient_quota`. O que os testes comprovam é que a falha é propagada
como `status=ERRO`, **sem** predição e **sem** sucesso aparente.

### 34.4 Integridade do repositório

- `models/modelo_pico_risco.pkl` — **não** modificado (mesmo SHA-256 antes/depois da correção);
- `models/modelo_pico_risco_metadata.json` — apenas `pasta_modelos` alterado (caminho relativo);
- `data/**` — **inalterado** (material oficial preservado);
- `notebooks/treinamento_modelo.ipynb` — **inalterado** nesta etapa;
- **nenhum** `commit` nem `push` executado (decisão do autor).

## 35. Limitações que seguem valendo após a Etapa 4

1. **Dados sintéticos**: o alvo é uma regra determinística reconstruível — as métricas altas medem
   aderência à regra, não desempenho clínico.
2. **Sem run real completo**: a condução dos handoffs por um LLM real permanece **não comprovada**
   (§34.3).
3. **Limiar 0,5 operacional**: sem análise de calibração nem custo de falso negativo.
4. **Conflito `gradio` × `pydantic`**: apenas documentado (§32).
5. **Placeholders de 0 byte**: mantidos (§33).
6. **Persistência**: cada execução tem contexto próprio (sem `session`/`conversation_id`).
7. **Tracing desativado**: sem observabilidade remota do SDK.
