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

- **Fase 6:**

---

## 1. Objetivo

Nessa sexta fase do **CardioIA** treinamos um modelo de classificação para prever a variável
pico_risco a partir de uma base de dados sintética e usar esse modelo dentro de um Sistema
multiagentes com quatro agentes especializados, orquestrados pelo **OpenAI Agents SDK** com
handoffs.

Solução:

1. **Machine Learning**: geração da base, treino e avaliação (acurácia, matriz de
   confusão, relatório de classificação);
2. **Protocolos determinísticos**: as 10 regras da base de conhecimento propostos são aplicadas em
   Python, com os limiares e operadores exatos do material — **nenhum LLM escolhe protocolo ou
   altera limite**;
3. **Multiagente**: quatro agentes com responsabilidades separadas, *handoffs*, *tools*
   que consultam o modelo e os protocolos, contexto tipado compartilhado e um contrato de saída;
4. **Interface de execução**: Recebe um paciente e devolve a resposta estruturada
   (texto legível ou JSON).


Algumas observações:

- Os dados são **sintéticos**, gerados por regras em `data/base_dados.py` (`np.random.seed(42)`,
  5.000 registros). **
- O alvo `pico_risco` é **construído deterministicamente** a partir de 6 indicadores booleanos
  (`risk_score >= 2`). Um modelo que reconstrói essa regra obtém métricas altíssimas.


## 2. Base de Dados e Treinamento


`data/base_dados.py` gera 5.000 registros com 10 variáveis e constrói o alvo por regra:

```text
risk_score = (carga_sistema > 80) + (tempo_espera > 60) + (spo2_media < 92)
           + (fc_media > 110) + (eventos_24h > 4) + (disponibilidade_recursos < 40)
pico_risco = 1 se risk_score >= 2, senão 0
```

Distribuição: **1.329 (26,58%)** positivos e 3.671 (73,42%) negativos.
`risk_score` **nunca** é usado como variável de entrada (seria vazamento do alvo) — a checagem é
feita por `assert` no notebook e em `dataset.separar_features_e_alvo`.

### Modelo

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


### Resultados do Machine Learning 

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


## 4. Agentes e Protocolos

| # | Agente | Responsabilidade | O que ele **não** faz |
|---|---|---|---|
| 1 | **Analista de Risco** | chamar a *tool* que consulta o modelo salvo e devolver a probabilidade real | não estima/arredonda probabilidade; não treina |
| 2 | **Especialista em Protocolos** | chamar a *tool* que aplica as 10 regras oficiais | não cria regra, não altera limite, não sugere protocolo fora da base |
| 3 | **Planejador de Resposta** | consolidar narrativa, justificativas e pontos de atenção (saída estruturada) | não recalcula nada; não introduz números |
| 4 | **Orquestrador** | validar a entrada, exigir o modelo, iniciar o run, montar/validar o contrato e registrar status/erros | não responde ao usuário por conta própria |

As instruções completas de cada agente estão nos respectivos módulos em
[`src/cardioia/agents/`](src/cardioia/agents/).


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


### Protocolos (determinísticos)

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


## 5. Arquitetura (resumo)

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


## 6. Estrutura do repositório

```text
cardioia_Fase6/
├── data/                          
│   ├── base_dados.py              # base sintética
│   ├── base_conhecimento.py       # 10 protocolos 
│   └── novo_paciente.py          
├── docs/
│   ├── plano_implementacao.md     # decisões e histórico das 5 etapas
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
├── tests/                         # 290 testes automatizados (pytest)
├── requirements.txt               # versões fixadas (testadas neste ambiente)
└── .env.example                   # nomes das variáveis (sem valores)
```


## 7. Tecnologias e dependências

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

## 8. Pré-requisitos e instalação

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

## 9. Como executar o pipeline

### 9.1 Modo simulado (padrão)

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

### 9.2 Modo real (API da OpenAI)

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

### 9.3 Códigos de saída

| Código | Significado |
|---|---|
| `0` | Pipeline concluiu (`CONCLUIDO` ou `CONCLUIDO_COM_RESSALVAS`) |
| `1` | `ERRO`/`INCOMPLETO` — motivos impressos em `stderr` |
| `2` | Erro de uso da CLI (flag inválida) |


## 10. Testes

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

### 10.1 Testes de integração (consome API paga)

```bash
# somente com chave disponível — os testes são PULADOS sem ela
python -m pytest tests/test_integracao_real.py -m integracao
```

## 12. Documentação, relatórios e diagramas

| Documento | Conteúdo |
|---|---|
| [`docs/plano_implementacao.md`](docs/plano_implementacao.md) | Decisões técnicas e histórico das 5 etapas (inventário, inconsistências, validações reais, limitações, pendências) |
| [`docs/arquitetura.md`](docs/arquitetura.md) | Diagramas Mermaid: componentes, sequência, fluxo de dados, fronteira de confiança e tratamento de falhas |
| [`docs/relatorio_tecnico.md`](docs/relatorio_tecnico.md) | Relatório técnico (problema, metodologia, resultados, limitações, conclusão) |
| [`docs/relatorio_tecnico.pdf`](docs/relatorio_tecnico.pdf) | Versão PDF do relatório (até 2 páginas) |
| [`docs/arquitetura_multiagente.pdf`](docs/arquitetura_multiagente.pdf) | PDF da arquitetura multiagente (até 3 páginas) |
