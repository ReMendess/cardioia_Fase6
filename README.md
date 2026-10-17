# AIReport Gen-Experience

<p align="center">
  <a href="https://www.fiap.com.br/"><img src="assets/logo-fiap.png" alt="FIAP" width="30%"></a>
</p>

<p align="center">
  <strong>Enterprise Challenge · Sprint 4 (fase final) · Dasa / Genera</strong><br>
  RAG + LLM + App React Native + Governança de IA
</p>

## Autor

| Nome | RM |
|---|---|
| [Renan de Oliveira Mendes](https://www.linkedin.com/in/renanmendes26/) | RM563145 |


## Vídeo

- **Sprint 4 (fase final):** https://youtu.be/tbgaDIVSIdc
- **App (fase final):** https://youtube.com/shorts/xfmcfVPZYBU?feature=share

---

## Visão geral

O **AIReport Gen-Experience** transforma o relatório genético (PDF) em uma experiência
conversacional clara, visual e segura — do PDF ao celular:

```text
React Native (Expo)  →  API REST (FastAPI)  →  RAG  →  ChromaDB  →  LLM (Groq)
```

O usuário conversa com o próprio relatório, vê cards de risco e ancestralidade em
linguagem simples, com **guardrails**, **LGPD programática**, **logs auditáveis**,
**avaliação de qualidade com evidências** e **operação monitorada** (status/métricas).

<img src="/assets/home_page.png" widht="100%">

> O assistente **não** emite diagnóstico, prescrição ou recomendações fora do
> relatório, e não substitui consulta médica.



### Fluxo de processamento
O fluxo principal da aplicação é:


- O usuário acessa o aplicativo e interage com a tela de consentimento.
- O aplicativo solicita os dados do relatório ou envia uma pergunta à API.
- A API verifica as condições de processamento e encaminha a solicitação ao serviço correspondente.
- Para perguntas, o sistema aplica validações de entrada e verifica se a solicitação está dentro do escopo.
- O mecanismo RAG realiza uma busca semântica no ChromaDB e recupera trechos relevantes.
- O modelo de linguagem recebe a pergunta e o contexto recuperado para gerar uma resposta.
- A resposta passa pelos mecanismos de validação e pelas políticas de saída.
- A API retorna o resultado ao aplicativo, incluindo fontes e informações de recusa quando aplicável.
- Eventos técnicos e métricas são registrados sem armazenar o conteúdo das perguntas, respostas ou dados genéticos nos logs.


## Agente Conversacional

O agente foi projetado para atuar como:

- Assistente interpretativo de relatórios genéticos.
- Ferramenta de simplificação de linguagem técnica.
- Recurso informativo.
- Interface conversacional para consulta de informações presentes no relatório.

<img src="/assets/resposta_agente.png" widht="80%">

## Aplicação Mobile

A aplicação foi desenvolvida seguindo o mesmo estilo que a versão web. Contendo:

- Cards informativos
- Botões de expansão, para mostrar mais informações por tópicos
- Chat com o agente conversacional
- Layout e visual responsivo para cada aparelho mobile

<img src="/assets/app_final.png" widht="70%">

## Governança e LGPD

Foram incorporadas verificações programáticas antes e depois da geração.

As instruções do agente orientam o modelo a:

- Utilizar linguagem clara e acessível.
- Manter uma comunicação não alarmista.
- Responder com base no contexto recuperado.
- Informar quando o relatório não contém os dados necessários.
- Evitar diagnósticos e recomendações terapêuticas.
- Preservar o caráter informativo das respostas.

O sistema possui mecanismos para identificar solicitações que não devem seguir para geração.
Entre os casos tratados estão:

- Tentativas de prompt injection.
- Perguntas fora do escopo do relatório.
- Solicitações sem contexto recuperável.
- Entradas que excedem os limites definidos.

Quando uma solicitação é recusada por não possuir contexto suficiente ou por violar as regras de entrada, o sistema retorna uma resposta de recusa com o motivo.

A aplicação disponibiliza um mecanismo de consentimento por meio da API e da interface mobile.
O consentimento pode ser consultado e registrado pelos endpoints correspondentes. 

<img src="/assets/consentimento.png" widht="100%">

O projeto implementa uma política de retenção de 30 dias para os registros abrangidos pelo mecanismo de expurgo de logs.
Também existe um recurso de expurgo executável por meio dos módulos de governança.
O histórico de conversa é tratado como contexto temporário de sessão, sem uma política de armazenamento permanente de conversas implementada como funcionalidade do protótipo.


## Testes e Validação

### Avaliação e qualidade do modelo

A Sprint 4 incorporou um processo reproduzível de avaliação do agente, com um conjunto de 18 perguntas de referência.
O objetivo é verificar o comportamento do sistema em situações representativas, incluindo perguntas sobre o relatório, solicitações sem contexto suficiente e casos que exigem aplicação das regras de segurança.

<img src="/assets/avaliacao_modelo.png" widht="100%">
<img src="/assets/validacao.png" widht="100%">


A aplicação também possui mecanismos para acompanhar a saúde dos serviços e identificar falhas operacionais.

<img src="/assets/validacao_pipeline.png" widht="150">


### Critérios avaliados

| Critério | Descrição |
|---|---|
| Aderência ao contexto | Verificar se a resposta respeita as informações disponíveis |
| Consistência | Avaliar a estabilidade das respostas em execuções repetidas |
| Groundedness | Medir a fundamentação textual das respostas no contexto recuperado |
| Violações | Identificar descumprimentos das regras definidas |
| Recusas | Verificar o tratamento de solicitações fora do escopo |

### Resultados da avaliação

Na avaliação registrada, foram obtidos os seguintes resultados:

| Métrica | Resultado |
|---|---:|
| Aderência | 1,00 |
| Consistência | 1,00 |
| Groundedness | Aproximadamente 0,571 |
| Violações identificadas | 0 |
| Veredito do harness | APROVADO |

Os resultados são referentes ao conjunto de testes, às configurações e às execuções documentadas. Não representam uma garantia de desempenho equivalente em todos os relatórios, usuários ou cenários de utilização.



<img src="/assets/resultado_testes.png" widht="150">


A Sprint 4 consolidou o AIReport Gen-Experience como uma aplicação com arquitetura mobile e API, incorporando controles técnicos e operacionais para tornar o uso da inteligência artificial mais rastreável, avaliável e responsável.

Os principais avanços foram:

- Integração entre React Native/Expo e FastAPI.
- Separação entre interface e processamento.
- Validação de entradas e respostas.
- Mecanismos de consentimento, retenção e expurgo.
- Logging estruturado com rastreabilidade.
-	Monitoramento por endpoints de saúde e métricas.
- Avaliação reproduzível do agente com conjunto de perguntas de referência.
-	Documentação de governança, riscos e limitações.

Na validação registrada, a solução apresentou 109 testes automatizados aprovados, 12 de 12 asserções no fluxo E2E, bundles Android e Web gerados com sucesso e avaliação do modelo aprovada pelos critérios definidos no harness.

<img src="/assets/status_sistema.png" widht="100%">

### Status da Sprint 4

| Bloco de requisito | Evidência |
|---|---|
| Governança de IA (LGPD, explicabilidade e logging) | `docs/politica_governanca_sprint4.md` · `logs/audit.log` |
| Operação e automação (monitoramento e falhas) | `/status` · `/metrics` · `scripts/monitor.py` |
| Avaliação do modelo (qualidade e consistência) | `docs/evidencia_evaluacion_sprint4.md` |
| Validação das respostas (clareza, aderência e recusa) | `app/validation/**` · 109 testes aprovados |
| Deploy (app React Native e fluxo ponta a ponta) | `mobile/**` · bundles Android/Web · E2E 12/12 |
| Refinamento final (consolidação, UX e performance) | Migração para `langchain-chroma` · paridade dos cards entre RN e Web |
| Documentação | `README.md` · `docs/**` · `mobile/README.md` |
---

## Evolução das 4 Sprints

| Sprint | Foco | Entregas |
|---|---|---|
| **2** | RAG conversacional | Ingestão do PDF (PyMuPDF), limpeza, embeddings (FastEmbed), ChromaDB, busca semântica, LLM via Groq, agente com memória, guardrails de prompt |
| **3** | Experiência visual | Dashboard, cards de risco, ancestralidade, linguagem simplificada (NLP), resumos automáticos, contexto de sessão, tom não alarmista, disclaimers |
| **4** | Produção e governança | API REST (8 endpoints), logging estruturado + auditoria (`trace_id`), LGPD (consentimento/retenção/expurgo), validações programáticas (injeção/off-topic/groundedness/refusal), harness de avaliação com evidências, operação (`/status`, `/metrics`, monitor), **app React Native (Expo)**, CORS e integração ponta a ponta |

---

## Arquitetura

```text
┌──────────────────────────────┐
│  App React Native (Expo)     │  mobile/
│  Consentimento · Painel ·    │
│  Chat · Status               │
└──────────────┬───────────────┘
               │ HTTP/JSON (CORS)
┌──────────────▼───────────────┐
│  API REST (FastAPI)          │  api/
│  /health /status /report     │
│  /chat /reprocess /consent   │
│  /metrics                    │
└──────────────┬───────────────┘
               │ reutiliza app/ (sem reescrever o RAG)
┌──────────────▼───────────────┐
│  Rag Engine + Validações     │  app/rag_engine.py · app/validation/
│  entrada · relevância ·      │
│  política de saída ·         │
│  groundedness                │
└──────────────┬───────────────┘
   ┌───────────┴───────────┐
   ▼                       ▼
┌─────────────────┐   ┌──────────────────┐
│  ChromaDB       │   │  LLM (Groq)      │
│  (langchain-    │   │  gpt-oss-120b    │
│   chroma)       │   │  temperatura 0.2 │
└─────────────────┘   └──────────────────┘

Camadas transversais (Sprint 4):
  Logging       → app/observability/logging_config.py (JSON + trace_id)
  Monitoring    → /status · /metrics · scripts/monitor.py
  Evaluation    → app/evaluation/harness.py + data/eval/questions.json
  Validation    → app/validation/** (entrada, groundedness, output_policy)
  Governance    → app/governance/** (consentimento, retenção, política)
  Security      → sanitização de entrada, CORS configurável, segredos em .env
```
---

## Funcionalidades

| Funcionalidade | Descrição |
|---|---|
| Leitura automática do PDF | Carrega `data/raw/*.pdf` sem upload manual (PyMuPDF + pdfplumber) |
| Base vetorial com cache | ChromaDB persistido; reindexa só se o PDF mudar (fingerprint) |
| Dashboard | Resumo (3 cards com dica), cards de risco ordenados por severidade, ancestralidade |
| Chat RAG | Perguntas em linguagem natural, respostas só com o contexto recuperado, **fontes exibidas** |
| Resumo automático por card | LLM gera resumo fundamentado no trecho do relatório |
| Validações programáticas | Bloqueio de injeção de prompt, off-topic e respostas sem fundamento (`refused` + motivo) |
| Groundedness | Score de fundamentação por resposta (exibido no app e no `/metrics`) |
| Governança LGPD | Consentimento (art. 8º), retenção de logs configurável, expurgo, política viva |
| Auditoria | `logs/audit.log` com metadados (nunca conteúdo do usuário) e `trace_id` por requisição |
| Operação | `/status` (PDF/índice/consentimento), `/metrics` (latências p95, recusas), monitor CLI |
| Avaliação do modelo | Set dourado de 18 perguntas + métricas + evidências versionadas |
| App móvel | Expo (Android/iOS/Web): consentimento → painel → chat → status |

## Tecnologias

| Camada | Tecnologia |
|---|---|
| App móvel | React Native (Expo SDK 51) + TypeScript + React Navigation |
| API | FastAPI + Uvicorn (CORS configurável) |
| Orquestração IA | LangChain |
| LLM | Groq — `openai/gpt-oss-120b` (temperatura 0.2) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (FastEmbed) |
| Banco vetorial | ChromaDB (via `langchain-chroma`) |
| PDF | PyMuPDF (`fitz`) + pdfplumber |
| Interface web (legada) | Streamlit |
| Linguagem | Python 3.10+ (validado em 3.13) · Node 18+ |

## Estrutura do repositório

```text
genreport-ai_Sprint3/
├── api/                      # API REST (FastAPI)
│   ├── main.py               # app + CORS + logging
│   ├── middleware.py         # trace_id + log de requisições
│   ├── schemas.py            # contratos (Pydantic)
│   └── routers/              # chat, report, consent, ops
├── app/                      # núcleo Python (reutilizado pela API e Streamlit)
│   ├── rag_engine.py         # RAG + validações
│   ├── validation/           # entrada, relevância, groundedness, política, refusals
│   ├── governance/           # política LGPD, consentimento, retenção/expurgo
│   ├── observability/        # logging JSON, auditoria, métricas
│   ├── evaluation/           # harness da avaliação
│   ├── prompt_engineering.py # prompts + guardrails
│   ├── risk_classifier.py / risk_card.py / report_parser.py / dashboard.py / ui.py
│   └── ...                   # parser_pdf, text_cleaner, embeddings, vector_store, summarizer, nlp_simplifier
├── mobile/                   # App React Native (Expo) — ver mobile/README.md
├── scripts/
│   ├── eval_run.py           # avaliação do modelo → docs/evidencia_*.md
│   └── monitor.py            # saúde do pipeline (exit 0/1)
├── data/
│   ├── raw/                  # PDF do relatório (fora do git — dado pessoal)
│   ├── vectordb/             # índice persistido (fora do git)
│   ├── eval/questions.json   # set dourado (sem PII)
│   └── consent.json          # consentimento (fora do git)
├── docs/
│   ├── politica_governanca_sprint4.md
│   ├── evidencia_evaluacion_sprint4.md
│   └── evidencia_iteracao1.md
├── logs/                     # app.log + audit.log (fora do git)
├── tests/                    # 109 testes (unittest)
├── streamlit_app.py          # interface web legada
├── requirements.txt          # versões fixadas
└── .env.example / .env
```

## Pré-requisitos

1. **Python 3.10+** (validado em 3.13) e **Node.js 18+** (para o app).
2. **Chave Groq** gratuita em [console.groq.com](https://console.groq.com/).
3. Relatório genético em `data/raw/genetic_report.pdf` (o projeto usa um simulado).

Crie o `.env` na raiz:

```bash
GROQ_API_KEY=gsk_sua_chave_aqui
GROQ_MODEL=openai/gpt-oss-120b
CORS_ALLOW_ORIGINS=*
```

## Como rodar

### 1) Backend + API REST (porta 8010)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |  Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt

python -m uvicorn api.main:app --host 127.0.0.1 --port 8010
# Documentação interativa: http://127.0.0.1:8010/docs
```

> Para o celular acessar, suba com `--host 0.0.0.0` e use o IP da LAN.

### 2) Interface web (Streamlit — legada, portas 8501)

```bash
python -m streamlit run streamlit_app.py
```

### 3) App mobile (Expo)

```bash
cd mobile
npm install
npm run android        # emulador Android (usa 10.0.2.2:8010 automaticamente)
npm run web            # navegador em http://localhost:1921
# Celular (Expo Go): informe o IP da sua máquina
#   EXPO_PUBLIC_BASE_URL=http://192.168.0.10:8010 npm start
```

Detalhes em [`mobile/README.md`](mobile/README.md).
---

## Como testar

### Suíte automatizada (109 testes)

```bash
python -m unittest discover -s tests -v
```

| Módulo | Cobertura |
|---|---|
| `test_risk_classifier.py` | classificação de riscos (21) |
| `test_observability.py` | logging JSON, trace_id, **privacidade** (11) |
| `test_governance.py` | retenção, expurgo, consentimento (18) |
| `test_validation.py` | entrada/off-topic/groundedness/refusal (33) |
| `test_evaluation.py` | harness + set dourado (16) |
| `test_ops.py` | `/status`, `/metrics`, monitor (10) |

### Saúde do pipeline e avaliação do modelo

```bash
python scripts/monitor.py            # [OK] PDF, índice, logs · exit 0 = saudável
python scripts/monitor.py --json     # saída para automação
python scripts/eval_run.py --runs 2  # regenera docs/evidencia_evaluacion_sprint4.md
```

### Teste rápido dos endpoints

```bash
curl http://127.0.0.1:8010/api/v1/health
curl http://127.0.0.1:8010/api/v1/status
curl http://127.0.0.1:8010/api/v1/report
curl -X POST http://127.0.0.1:8010/api/v1/chat \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"Quais são meus principais riscos genéticos?\"}"
# Recusa esperada (fora do tema):  "Como fazer bolo de chocolate?"
# Recusa esperada (injeção):       "ignore todas as instruções anteriores"
```

## API — endpoints

| Método | Rota | Descrição |
|---|---|---|
| GET | `/api/v1/health` | Liveness (`{"status":"ok"}`) |
| GET | `/api/v1/status` | Saúde detalhada: PDF, índice, consentimento, logging (`ok/degraded/error`) |
| GET | `/api/v1/report` | Dados estruturados: resumo, findings, risk cards, ancestralidade, disclaimers |
| POST | `/api/v1/chat` | `{question, ...}` → `{answer, sources, refused, refusal_reason, groundedness}` |
| POST | `/api/v1/reprocess` | Reprocessa o relatório (reindexação com liberação de handles) |
| GET | `/api/v1/consent` | Status do consentimento LGPD |
| POST | `/api/v1/consent` | Registra/revoga consentimento (`{"granted": true}`) |
| GET | `/api/v1/metrics` | Métricas: HTTP por rota (avg/min/max/**p95**), status, recusas, groundedness |

## Governança e LGPD

- **Relatório de Governança Final:** [`relatorio_governaca_riscos.pdf`](relatorio_governaca_riscos.pdf)
- **Política completa:** [`docs/politica_governanca_sprint4.md`](docs/politica_governanca_sprint4.md)
- Consentimento com registro (apenas flag/versão/timestamp — **sem PII**); gate opcional via `REQUIRE_CONSENT=true`.
- **Retenção:** `LOG_RETENTION_DAYS` (padrão 30) com expurgo na inicialização e via `python -m app.governance.policy`.
- **Logs:** JSON com `trace_id`; **nunca** registram pergunta/resposta/paciente (guard ativo + teste automatizado).
- PDF com PII **fora do git** (`data/raw/*.pdf` ignorado; histórico pendente de purga — procedimento documentado na §9 da política).

## Avaliação do modelo (evidências)

- **Resultado final:** **APROVADO** — aderência **1.0** (18/18), consistência **1.0**, groundedness média **0.571**, **0 violações** de política.
- Documentos: [`docs/evidencia_evaluacion_sprint4.md`](docs/evidencia_evaluacion_sprint4.md) e iteração registrada em [`docs/evidencia_iteracao1.md`](docs/evidencia_iteracao1.md).

## Operação e monitoramento

- `/status` alimenta o app (banner degradado + botão de reprocessar).
- `/metrics` expõe latências p95 por rota, códigos HTTP, turnos de chat, recusas por motivo e groundedness médio.
- `scripts/monitor.py` é o health-check para cron/CI (exit code 0/1).
- Auditoria em `logs/audit.log` (eventos `chat_turn`, `report_access`, `reprocess`, `consent_update`).

## Deploy

```bash
# Terminal 1 — API acessível na rede
python -m uvicorn api.main:app --host 0.0.0.0 --port 8010
# Terminal 2 — App
cd mobile && npm run android     # ou npm run web / Expo Go com EXPO_PUBLIC_BASE_URL
```


## User stories

| ID | História | Status |
|----|----------|--------|
| US1 | Entender o exame em linguagem simples | ✅ chat + resumos + linguagem simplificada |
| US2 | Fazer perguntas sobre meu exame | ✅ chat RAG com fontes |
| US3 | Ver resumo dos principais riscos | ✅ dashboard com cards priorizados |
| US4 | Usar no celular | ✅ app React Native (Expo) |
| US5 | Confiar no uso dos meus dados | ✅ consentimento, política, auditoria e recusas |
