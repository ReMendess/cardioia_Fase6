# CardioIA — Fase 6

<p align="center">
  <a href="https://www.fiap.com.br/"><img src="assets/logo-fiap.png" alt="FIAP" width="30%"></a>
</p>

<p align="center">
  <strong>Sistema preditivo multiagente para análise de risco</strong><br>
  scikit-learn (modelo supervisionado) + base de protocolos determinística + OpenAI Agents SDK
</p>

---

## Autor

| Nome | RM |
|---|---|
| [Renan de Oliveira Mendes](https://www.linkedin.com/in/renanmendes26/) | RM563145 |

## Vídeo

- **Fase 6:** _(a fornecer)_

## Repositório

- https://github.com/ReMendess/cardioia_Fase6

---

## 1. Objetivo

O **CardioIA Fase 6** treina um classificador supervisionado para prever a variável
`pico_risco` a partir de uma base **sintética** e usa esse modelo dentro de um **sistema
multiagente** com quatro agentes especializados, orquestrados pelo **OpenAI Agents SDK** com
*handoffs* reais.

O que a entrega demonstra, de ponta a ponta:

1. **Machine Learning**: geração/inspeção da base, treino, avaliação (acurácia, matriz de
   confusão, relatório de classificação) e persistência do modelo;
2. **Protocolos determinísticos**: as 10 regras da base de conhecimento da aula são aplicadas em
   Python, com os limiares e operadores exatos do material — **nenhum LLM escolhe protocolo ou
   altera limite**;
3. **Multiagente**: quatro agentes com responsabilidades separadas, *handoffs* do SDK, *tools*
   que consultam o modelo e os protocolos, contexto tipado compartilhado e um contrato de saída
   validado com Pydantic;
4. **Interface de execução**: CLI que recebe um paciente e devolve a resposta estruturada
   (texto legível ou JSON), com códigos de saída explícitos.

## 2. Escopo acadêmico e limitações do uso de dados sintéticos

- Os dados são **sintéticos**, gerados por regras em `data/base_dados.py` (`np.random.seed(42)`,
  5.000 registros). **Nenhum dado de paciente real foi utilizado.**
- O alvo `pico_risco` é **construído deterministicamente** a partir de 6 indicadores booleanos
  (`risk_score >= 2`). Um modelo que reconstrói essa regra obtém métricas altíssimas —
  isso mede **aderência à regra**, não capacidade preditiva em cenário real.
- Os limiares dos protocolos são de **projeto** (materiais da aula) e não representam conduta
  clínica.
- O limiar de decisão (0,5) é **operacional**, escolhido por coincidir com a decisão padrão de
  um classificador binário; **não** foi calibrado com dados reais.
- Este repositório é um **artefato acadêmico** (FIAP · Enterprise Challenge). Não produz
  diagnóstico, prescrição nem substitui avaliação profissional de saúde.

## 3. Arquitetura (resumo)

```text
CLI (python -m src.cardioia.main)
  │  valida o paciente  ·  exige models/modelo_pico_risco.pkl (nunca treina aqui)
  ▼
Orquestrador  ──────────────────────────  Runner.run_sync(context=ContextoDaExecucao)
  │
  ├─ handoff ─▶ Analista de Risco ── tool ─▶ cardioia.ml.predict  (probabilidade real)
  ├─ handoff ─▶ Especialista em Protocolos ── tool ─▶ cardioia.knowledge.protocols
  └─ handoff ─▶ Planejador de Resposta (output_type estruturado: narrativa em texto)
  │
  ▼  (em Python, depois do run)
RespostaDaExecucao (contrato Pydantic) montada a partir do CONTEXTO tipado
```

Detalhes, diagramas Mermaid e tratamento de falhas: [`docs/arquitetura.md`](docs/arquitetura.md).

**Ponto central de confiança:** números (probabilidade, classe, protocolos, justificativas) são
calculados por **Python** e gravados num **contexto tipado** (`RunContextWrapper.context`), que a
documentação do SDK define como objeto local **nunca enviado ao LLM**. O LLM redige a narrativa;
se ela citar um número divergente, o pipeline marca `CONCLUIDO_COM_RESSALVAS` e o valor
autoritativo continua sendo o da *tool*.

## 4. Estrutura do repositório

```text
cardioia_Fase6/
├── data/                          
│   ├── base_dados.py              # base sintética
│   ├── base_conhecimento.py       # 10 protocolos 
│   └── novo_paciente.py          
├── docs/
│   ├── plano_implementacao.md     # decisões e histórico das 4 etapas
│   ├── arquitetura.md             # diagramas Mermaid (componentes, sequência, dados)
│   ├── relatorio_tecnico.md       # relatório técnico
│   ├── relatorio_tecnico.pdf      # versão PDF do relatório
│   └── arquitetura_multiagente.pdf# PDF da arquitetura multiagente
├── models/
│   ├── modelo_pico_risco.pkl          # modelo treinado (reconstruível)
│   └── modelo_pico_risco_metadata.json# rastreabilidade: features, métricas, versões, SHA-256
├── notebooks/
│   └── treinamento_modelo.ipynb       # ML com explicações e gráficos de texto
├── src/cardioia/                  # IMPLEMENTAÇÃO OFICIAL
│   ├── config.py                  # constantes compartilhadas (artefatos, limiar, turnos)
│   ├── main.py                    # CLI
│   ├── ml/                        # dataset.py · train.py · evaluate.py · predict.py
│   ├── knowledge/protocols.py     # protocolos oficiais + regras determinísticas
│   ├── schemas/risk_result.py     # contrato Pydantic de entrada/saída
│   └── agents/                    # contexto.py · 4 agentes · modelo_simulado.py
├── tests/                         # 281 testes automatizados (pytest)
├── requirements.txt               # versões fixadas (testadas neste ambiente)
└── .env.example                   # nomes das variáveis (sem valores)
```

## 5. Tecnologias e dependências

| Camada | Tecnologia | Versão testada |
|---|---|---|
| Linguagem | Python | 3.13.5 |
| Dados | pandas · NumPy | 2.3.2 · 2.2.6 |
| Machine Learning | scikit-learn | 1.8.0 |
| Persistência do modelo | joblib | 1.5.3 |
| Multiagente | **openai-agents** (OpenAI Agents SDK) | **0.18.1** |
| SDK base | openai | 2.38.0 |
| Contratos | pydantic | 2.14.1 |
| Testes | pytest | 9.1.1 |

Tudo está fixado em [`requirements.txt`](requirements.txt). A versão do SDK foi escolhida por
**compatibilidade real de dependências** (a 0.18.2+ exigiria `openai>=2.45`; a 0.22+ exigiria
`openai` 3.x). O arquivo também registra um **conflito de ambiente conhecido**: `gradio 5.49.1`
declara `pydantic<2.12`, enquanto o SDK exige `pydantic>=2.12.2` — restrições mutuamente
insatisfazíveis (detalhes e alternativa na §14).

## 6. Pré-requisitos e instalação

- **Python 3.10+** (validado em 3.13.5).
- **Git** (opcional, para clonar).
- Chave da OpenAI **apenas** se for usar o modo real (§8.4).

```bash
git clone https://github.com/ReMendess/cardioia_Fase6.git
cd cardioia_Fase6

python -m venv .venv
# Windows (PowerShell):  .venv\Scripts\Activate.ps1
# Linux/macOS:           source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Para o modo real, copie o modelo de variáveis e preencha **localmente**:

```bash
cp .env.example .env      # Windows: Copy-Item .env.example .env
# edite .env e informe OPENAI_API_KEY (o arquivo é ignorado pelo Git)
```

> O código **não** carrega `.env` automaticamente: `OPENAI_API_KEY` é lida do ambiente. No
> PowerShell, exporte antes de rodar: `$env:OPENAI_API_KEY = "sk-..."`.

## 7. Como gerar ou atualizar o modelo

O treino é determinístico (semente 42) e grava os artefatos em `models/`:

```bash
# treina, avalia, escolhe pelo critério declarado e salva
python -m src.cardioia.ml.train

# apenas mede (não grava nada) — confere métricas sem tocar nos artefatos
python -m src.cardioia.ml.train --dry-run

# opções disponíveis
python -m src.cardioia.ml.train --pasta-modelos models --semente 42 --tamanho-teste 0.2 \
                               --n-estimators 300 --quiet
```

| Artefato | Conteúdo |
|---|---|
| `models/modelo_pico_risco.pkl` | Modelo escolhido (Random Forest), serializado com joblib |
| `models/modelo_pico_risco_metadata.json` | Features e ordem, parâmetros, semente, split, métricas de teste, matriz de confusão, faixas observadas, versões das bibliotecas, SHA-256 da base e o aviso de dados sintéticos |

> O pipeline **nunca** treina silenciosamente: se o artefato não existir, a CLI falha com a
> instrução do comando de treino.

## 8. Como executar o pipeline

### 8.1 Modo simulado (padrão)

Demonstra o fluxo completo **sem consumir créditos de API**: o SDK, os *handoffs*, as *tools*, o
`RunHooks` e o contrato são **reais**; apenas a decisão de rota do LLM é substituída por um
roteiro determinístico
([`src/cardioia/agents/modelo_simulado.py`](src/cardioia/agents/modelo_simulado.py)).

```bash
python -m src.cardioia.main            # paciente de data/novo_paciente.py
python -m src.cardioia.main --quiet    # apenas o status final
```

Saída (trecho, paciente da aula):

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
  - Se carga do sistema > 85% ...         [valor observado: 92]
  - Se disponibilidade_recursos < 30% ... [valor observado: 25]
  - Se densidade_eventos > 6 ...          [valor observado: 7]
  - Se risco sistemico alto ...           [classe prevista = 1]

HANDOFFS REALIZADOS
  -> Orquestrador -> Analista de Risco
  -> Analista de Risco -> Especialista em Protocolos
  -> Especialista em Protocolos -> Planejador de Resposta
```

> **O modo simulado não equivale a uma execução bem-sucedida com LLM real.** Ele prova o
> encadeamento entre agentes, tools e contrato; **não** prova que um modelo de linguagem escolhe
> os *handoffs* corretamente — essa validação é a do §9.2.

### 8.2 Saída em JSON

```bash
python -m src.cardioia.main --json > resposta.json
```

O JSON é o contrato completo (`RespostaDaExecucao`): `id_execucao`, `status`, `modo`,
`diagnostico` (probabilidade **crua**, sem arredondamento), `indicadores`, `protocolos`,
`justificativas`, `narrativa`, `avisos`, `limitacoes`, `erros`, `passos`, `handoffs_realizados`,
`agente_final`, `versao_do_sdk` e `duracao_segundos`.

### 8.3 Paciente por arquivo JSON

```bash
python -m src.cardioia.main --paciente caso.json
```

Use os nomes da base gerada (`disponibilidade_recursos` com **underscore**; a grafia
`disponibilidade recursos`, com espaço, é aceita e normalizada):

```json
{
  "idade_media": 62, "carga_sistema": 92, "tempo_espera": 75, "fc_media": 118,
  "spo2_media": 89, "pa_sistolica": 100, "temperatura": 37.5,
  "densidade_eventos": 7, "eventos_24h": 5, "disponibilidade_recursos": 25
}
```

### 8.4 Modo real (API da OpenAI)

```bash
# Windows PowerShell
$env:OPENAI_API_KEY = "sk-..."
python -m src.cardioia.main --modo real --modelo gpt-4o-mini

# Linux/macOS
export OPENAI_API_KEY="sk-..."
python -m src.cardioia.main --modo real --modelo gpt-4o-mini
```

- Sem a chave, o pipeline responde `status=ERRO` com a instrução e sai com código 1.
- O *tracing* do SDK fica **desativado** (`tracing_disabled=True`).
- `OPENAI_MODEL` (variável de ambiente) é alternativa à flag `--modelo`.

### 8.5 Códigos de saída

| Código | Significado |
|---|---|
| `0` | Pipeline concluiu (`CONCLUIDO` ou `CONCLUIDO_COM_RESSALVAS`) |
| `1` | `ERRO`/`INCOMPLETO` — motivos impressos em `stderr` |
| `2` | Erro de uso da CLI (flag inválida) |

### 8.6 Flags da CLI

`--modo {simulado,real}` · `--paciente` · `--json` · `--raiz` · `--pasta-modelos` · `--limiar` ·
`--max-turnos` · `--quiet` · `--modelo`. Consulte `python -m src.cardioia.main --help`.

## 9. Testes

### 9.1 Suíte principal (não usa a API da OpenAI)

```bash
python -m pytest tests -q

# por arquivo
python -m pytest tests/test_protocols.py -q     # 61 testes
python -m pytest tests/test_schemas.py -q       # 61 testes
python -m pytest tests/test_prediction.py -q    # 40 testes
python -m pytest tests/test_dataset.py -q       # 33 testes
python -m pytest tests/test_orchestrator.py -q  # 29 testes
python -m pytest tests/test_agents.py -q        # 26 testes
python -m pytest tests/test_seguranca.py -q     # 17 testes
python -m pytest tests/test_cli.py -q           # 14 testes
```

| Arquivo | Cobre |
|---|---|
| `test_dataset.py` | base oficial, dimensões, alvo, `risk_score` fora das features, reprodutibilidade |
| `test_prediction.py` | contrato de entrada, probabilidade, modelo ausente, estabilidade após reload |
| `test_protocols.py` | as 10 regras, limites **estritos**, multiseleção, ausência de duplicatas, dados inválidos |
| `test_schemas.py` | contratos Pydantic: probabilidade, classes, campos obrigatórios, coerência de status |
| `test_agents.py` | tools chamando as implementações reais, configuração dos handoffs, contexto |
| `test_orchestrator.py` | fluxo completo com handoffs reais, histórico de mensagens, falhas |
| `test_cli.py` | comando real, `--json` válido, códigos de saída, flags, independência de diretório |
| `test_seguranca.py` | falha de API propagada, ausência de segredos, portabilidade dos caminhos, coerência dos protocolos com o arquivo oficial |

### 9.2 Testes de integração (opcional, consome API paga)

```bash
# somente com chave disponível — os testes são PULADOS sem ela
python -m pytest tests/test_integracao_real.py -m integracao
```

> **Estado real (registrado na Etapa 3):** a chave autentica e a requisição chega à API, mas a
> conta responde `HTTP 429 / insufficient_quota`. Logo, **não houve execução completa
> bem-sucedida com LLM real** e a condução dos *handoffs* pelo modelo de linguagem **não está
> comprovada**. O que os testes verificam nesse cenário é que a falha **não** é mascarada como
> sucesso (`status=ERRO`, sem predição). Nada disso é falha dos testes unitários.

### 9.3 Sonda de sanidade do artefato (opcional)

```bash
python -m src.cardioia.ml.train --dry-run   # mede sem gravar: LR 0,816 / RF 0,999
```

## 10. Os quatro agentes

| # | Agente | Responsabilidade | O que ele **não** faz |
|---|---|---|---|
| 1 | **Analista de Risco** | chamar a *tool* que consulta o modelo salvo e devolver a probabilidade real | não estima/arredonda probabilidade; não treina |
| 2 | **Especialista em Protocolos** | chamar a *tool* que aplica as 10 regras oficiais | não cria regra, não altera limite, não sugere protocolo fora da base |
| 3 | **Planejador de Resposta** | consolidar narrativa, justificativas e pontos de atenção (saída estruturada) | não recalcula nada; não introduz números |
| 4 | **Orquestrador** | validar a entrada, exigir o modelo, iniciar o run, montar/validar o contrato e registrar status/erros | não responde ao usuário por conta própria |

As instruções completas de cada agente estão nos respectivos módulos em
[`src/cardioia/agents/`](src/cardioia/agents/).

## 11. Handoffs e tools

**Handoffs (reais, do SDK):**

```text
Orquestrador ──▶ Analista de Risco ──▶ Especialista em Protocolos ──▶ Planejador de Resposta
```

O SDK os expõe ao LLM como ferramentas `transfer_to_<agente>` e registra cada passo em
`RunHooks.on_handoff` — o pipeline guarda a trilha observável em `handoffs_realizados` e
`passos`.

**Tools (Python, executadas pelo SDK):**

| Tool | Agente | Chama | Devolve ao modelo |
|---|---|---|---|
| `consultar_modelo_de_risco` | Analista de Risco | `cardioia.ml.predict.prever_pico_risco` | resumo textual com probabilidade, classe, modelo e limiar |
| `consultar_protocolos` | Especialista em Protocolos | `cardioia.knowledge.protocols.avaliar_protocolos_seguro` | protocolos aplicáveis **com** condição satisfeita, valor observado e origem |

> As duas *tools* são declaradas **sem parâmetros** de propósito: elas leem o paciente validado do
> contexto. Se as variáveis fossem argumentos, o LLM poderia alterar os valores. Assim ele decide
> **quando** consultar, nunca **com que números**.

## 12. Como são calculados o risco e os protocolos

### 12.1 O rótulo `pico_risco` (material oficial)

`data/base_dados.py` gera 5.000 registros com 10 variáveis e constrói o alvo por regra:

```text
risk_score = (carga_sistema > 80) + (tempo_espera > 60) + (spo2_media < 92)
           + (fc_media > 110) + (eventos_24h > 4) + (disponibilidade_recursos < 40)
pico_risco = 1 se risk_score >= 2, senão 0
```

Distribuição: **1.329 (26,58%)** positivos e 3.671 (73,42%) negativos.
`risk_score` **nunca** é usado como variável de entrada (seria vazamento do alvo) — a checagem é
feita por `assert` no notebook e em `dataset.separar_features_e_alvo`.

### 12.2 O modelo

- **features**: as 10 colunas originais, na ordem do arquivo oficial;
- **split**: 80/20 **estratificado**, `random_state=42` (4.000 treino / 1.000 teste);
- **algoritmos comparados** no **mesmo** treino e teste:
  `LogisticRegression` (pipeline com `StandardScaler`, interpretável) e
  `RandomForestClassifier(n_estimators=300, random_state=42)`;
- **critério de escolha, declarado antes de medir**: maior *recall* da classe `pico_risco=1`;
  empate por acurácia; depois por interpretabilidade. O conjunto de teste é usado **uma única
  vez**, para medir — não para ajustar hiperparâmetros;
- **limiar de decisão**: 0,5 (operacional; igual ao `predict` padrão do scikit-learn). A
  probabilidade é devolvida **crua** — o arredondamento ocorre só na apresentação.

### 12.3 Os protocolos (determinísticos)

As 10 regras de `data/base_conhecimento.py` são lidas e associadas, **por posição validada**, a
regras numéricas em Python (`src/cardioia/knowledge/protocols.py`):

| Protocolo | Condição implementada | Limiar |
|---|---|---|
| 1 | `spo2_media < 88` | 88 |
| 2 | `carga_sistema > 85` | 85 |
| 3 | `fc_media > 120` | 120 |
| 4 | `eventos_24h > 5` | 5 |
| 5 | `disponibilidade_recursos < 30` | 30 |
| 6 | `tempo_espera > 90` | 90 |
| 7 | `temperatura > 38` | 38 |
| 8 | `densidade_eventos > 6` | 6 |
| 9 | `pa_sistolica < 90` | 90 |
| 10 | `pico_risco_previsto == 1` (classe do **modelo de ML**) | — |

Propriedades garantidas por teste:

- **operadores estritos**: `spo2_media == 88` **não** dispara `spo2_media < 88`;
- **sem correspondência textual aproximada**: o vínculo protocolo↔regra é uma lista fixa,
  validada contra o texto oficial — se o arquivo mudar, o carregamento **falha** em vez de casar
  com a regra errada;
- **sem duplicatas** e **somente** os protocolos satisfeitos são retornados;
- cada protocolo retornado traz **condição satisfeita**, **valor observado** e **origem** do
  valor (dados do paciente ou classificação do modelo);
- `data/base_conhecimento.py` **não é modificado** e um teste compara os limiares implementados com
  os extraídos do próprio arquivo oficial.

> Os limiares dos protocolos **diferem** dos limiares do rótulo (ex.: protocolo usa
> `carga_sistema > 85`, o rótulo usa `> 80`). São camadas distintas: o modelo aprende o rótulo; os
> protocolos continuam sendo avaliados deterministicamente.

## 13. Como reproduzir o treinamento e avaliar o modelo

### 13.1 Pelo notebook (recomendado para a entrega)

[`notebooks/treinamento_modelo.ipynb`](notebooks/treinamento_modelo.ipynb) contém o experimento
completo com explicações: inspeção da base, split, treino, métricas, análise crítica, salvamento,
recarga do artefato e a simulação com o paciente da aula.

**Google Colab:** envie o notebook e a pasta do projeto (a célula de bootstrap localiza a raiz,
insere `src/` no `sys.path` e falha com instrução clara se o projeto não estiver no ambiente).

**Local:**

```bash
# gera notebooks/treinamento_modelo.nbconvert.ipynb (o padrao ".nbconvert.ipynb"
# ja esta no .gitignore: a saida executada nao entra no versionamento)
python -m nbconvert --to notebook --execute notebooks/treinamento_modelo.ipynb
```

Validação executada nesta etapa: o notebook rodou de ponta a ponta **sem erros** (34 células,
19 de código), reproduzindo as mesmas métricas (LR 0,816 / RF 0,999, recall 0,9962) e o caso do
paciente da aula.

### 13.2 Pela linha de comando

```bash
python -m src.cardioia.ml.train            # treina, avalia, salva
python -m src.cardioia.ml.train --dry-run  # mede sem gravar
```

### 13.3 Como verificar os resultados

1. `models/modelo_pico_risco_metadata.json` traz as métricas de teste, as matrizes de confusão e o
   `sha256_da_base_usada`;
2. `python -m src.cardioia.ml.train --dry-run` imprime o comparativo no mesmo conjunto de teste;
3. o notebook imprime as mesmas métricas e compara com o **teto** da regra determinística.

## 14. Resultados medidos e limitações metodológicas

### 14.1 Resultados do Machine Learning (reproduzidos no notebook)

Base: 5.000 registros · 10 variáveis · alvo com 26,58% de positivos.
Split: 80/20 estratificado, semente 42 → **4.000 treino / 1.000 teste** (266 positivos no teste).

| Modelo | Acurácia | Recall (classe 1) | Precisão (classe 1) | F1 (classe 1) | Matriz de confusão `[[TN, FP], [FN, TP]]` |
|---|---:|---:|---:|---:|---|
| Regressão Logística | 0,8160 | 0,5526 | 0,6934 | 0,6151 | `[[669, 65], [119, 147]]` |
| **Random Forest** (escolhido) | **0,9990** | **0,9962** | 1,0000 | 0,9981 | `[[734, 0], [1, 265]]` |
| Regra determinística (teto) | 1,0000 | 1,0000 | 1,0000 | 1,0000 | `[[734, 0], [0, 266]]` |

Fonte: `models/modelo_pico_risco_metadata.json` e o notebook executado. O critério declarado
(maior *recall* da classe 1) escolheu o **Random Forest**; o único falso negativo tem
`risk_score = 2`, exatamente na fronteira do corte do rótulo.

### 14.2 Leitura metodológica (o que esses números **não** dizem)

- O alvo é uma **função determinística** de 6 variáveis com limiares fixos. O Random Forest
  reconstrói essa regra — a acurácia de 0,999 mede **aderência à regra**, não generalização.
- A Regressão Logística (0,816) mostra que a fronteira **não** é linearmente separável nessas
  variáveis: a diferença entre os modelos é explicada pela forma do rótulo.
- O "teto" de 1,0 da regra determinística é **tautológico** (a regra define o próprio rótulo) e só
  serve de referência.
- **Nenhuma conclusão clínica** pode ser extraída daqui: dados sintéticos, sem pacientes reais,
  sem validação externa.
- As métricas valem para a semente 42 e as versões registradas no metadata.

### 14.3 O que foi verificado e o que **não** foi

| Tipo de evidência | Resultado |
|---|---|
| **ML** (treino e avaliação) | ✅ Medido e reproduzido (notebook + CLI + metadata) |
| **Testes de software** | ✅ **281 testes aprovados, 3 pulados** (`python -m pytest tests -q`) |
| **Execução do SDK com modelo simulado** | ✅ Handoffs, tools, contexto e contrato exercitados com o SDK real |
| **Execução bem-sucedida com LLM real** | ❌ **Não ocorreu**: a API respondeu `HTTP 429 / insufficient_quota`. A chave autentica e a requisição chega ao endpoint, mas não houve run completo; **não** se afirma que o LLM escolheu os handoffs corretamente |

### 14.4 Conflito de dependências conhecido (não resolvido)

| Pacote | Restrição | Situação |
|---|---|---|
| `openai-agents 0.18.1` | `pydantic>=2.12.2` | ✅ satisfeita (2.14.1) |
| `gradio 5.49.1` (outro projeto do ambiente) | `pydantic<2.12` | ❌ **incompatível** com o SDK fixado |

As duas restrições são **mutuamente insatisfazíveis**. `import gradio` continua funcionando, mas
o pacote fica fora da faixa que ele próprio declara. Alternativas verificadas:

1. usar `openai-agents==0.4.2` (exige `pydantic>=2.10` e `openai>=2.2,<3`) — SDK bem mais antigo;
2. manter ambientes separados (um `venv` por projeto) — recomendado para uso continuado.

**Não** se declara o conflito resolvido: ele está apenas documentado.

## 15. Documentação, relatórios e diagramas

| Documento | Conteúdo |
|---|---|
| [`docs/plano_implementacao.md`](docs/plano_implementacao.md) | Decisões técnicas e histórico das 4 etapas (inventário, inconsistências, validações reais, limitações, pendências) |
| [`docs/arquitetura.md`](docs/arquitetura.md) | Diagramas Mermaid: componentes, sequência, fluxo de dados, fronteira de confiança e tratamento de falhas |
| [`docs/relatorio_tecnico.md`](docs/relatorio_tecnico.md) | Relatório técnico (problema, metodologia, resultados, limitações, conclusão) |
| [`docs/relatorio_tecnico.pdf`](docs/relatorio_tecnico.pdf) | Versão PDF do relatório (até 2 páginas) |
| [`docs/arquitetura_multiagente.pdf`](docs/arquitetura_multiagente.pdf) | PDF da arquitetura multiagente (até 3 páginas) |

## 16. Limitações conhecidas e próximos passos

1. **Sem execução real completa**: ver §9.2 e §14.3. Os *handoffs* decididos por LLM permanecem
   não validados.
2. **Dados sintéticos**: ver §2 e §14.2.
3. **Conflito `gradio` × `pydantic`**: ver §14.4.
4. **Sem persistência entre execuções**: cada `run` cria contexto e histórico próprios (não há
   `session`/`conversation_id`).
5. **Tracing desativado** por padrão: sem observabilidade remota do SDK.
6. **Placeholders vazios** em `src/`: mantidos por serem parte da estrutura inicial da atividade;
   a implementação oficial é `src/cardioia/**`.

Próximos passos possíveis: executar o modo real com créditos disponíveis; avaliar `output_guardrails`
do SDK para a narrativa; centralizar mais configurações; e, se a disciplina exigir, remover os
placeholders (com autorização).