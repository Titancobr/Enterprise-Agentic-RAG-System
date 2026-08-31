# IP-SAKTI Sahayak

SIH 2026 PS-26045 prototype for Ayurveda intellectual-property and regulatory guidance.

IP-SAKTI Sahayak is an auditable, source-grounded RAG assistant that helps users ask about Ayurveda patents, GI, traditional knowledge, ABS/biodiversity compliance, FSSAI/AYUSH pathways, and formulation classification.

## Live MVP Architecture

```text
User
→ Streamlit Chat UI
→ FastAPI Backend
→ Input Guardrails
→ Query + Jurisdiction Router
→ LangGraph State
→ Hybrid RAG: BM25 + Qdrant
→ FlashRank Reranker
→ Authorized Evidence Filter
→ Portkey/Groq Primary + Fallback Model
→ Citation & Output Verification
→ Final Answer + Trust Panel
```

Fine-tuning is kept as optional Phase 2 scaffolding. The live MVP is optimized for reliability, demo speed, and measurable quality.

## Repository Layout

```text
RAG_enterPriseSystem/
  app/                 FastAPI backend, LangGraph agent, guardrails, retrieval
  ui/                  Streamlit chat interface
  evals/               RAGAS and guardrail evaluation pipeline
  DATA/legal_corpus/   Authorized legal/regulatory prototype corpus
  assets/              SIH slides, architecture diagrams, slide previews
  DOCS/                Supporting technical documentation
```

## Quick Start

```bash
cd RAG_enterPriseSystem
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add keys in `RAG_enterPriseSystem/.env`, then start the backend:

```bash
uvicorn app.main:app --reload --port 8000
```

Start the UI in another terminal:

```bash
cd RAG_enterPriseSystem/ui
streamlit run app.py
```

## Evaluation

```bash
cd RAG_enterPriseSystem/evals
streamlit run app.py
```

Metrics include faithfulness, context precision, context recall, answer relevancy, citation correctness, guardrail behavior, jurisdiction accuracy, and ABS classification checks.

## SIH Deliverables

- Main project docs: `RAG_enterPriseSystem/README.md`
- Architecture: `RAG_enterPriseSystem/ARCHITECTURE.md`
- MVP checklist: `RAG_enterPriseSystem/MVP_READINESS_CHECKLIST.md`
- Source manifest: `RAG_enterPriseSystem/SOURCES_MANIFEST.md`
- Presentation assets: `RAG_enterPriseSystem/assets/`

## Safety Note

This prototype provides general legal/regulatory information only. It is not legal advice and should be reviewed by qualified legal or regulatory experts before production use.
