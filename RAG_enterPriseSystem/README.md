# IP-SAKTI Sahayak — Prototype (SIH 2026, PS-26045)

Auditable, multilingual, source-cited AI assistant for Ayurveda IP & regulatory guidance.

## Architecture Overview

```
User Query
    ↓
Language Detection (Bhashini/IndicTrans2)
    ↓
Input Guardrails (NeMo)
    ├─ Prompt Injection Check
    ├─ Scope Check (Ayurveda/IP only)
    └─ Jailbreak Protection
    ↓
Query Classifier (Planner)
    ├─ REGULATORY_IP → Jurisdiction Router → Hybrid Retrieval → Portkey/Groq LLM
    ├─ FORMULATION_CLASSIFICATION → Classifier → Jurisdiction Router → Retrieval → LLM
    ├─ CONVERSATIONAL → Direct Response (memory)
    └─ OFF_TOPIC → Refuse
    ↓
Output Guardrails
    ├─ Citation Verification
    ├─ Hallucination Check
    ├─ Jurisdiction Consistency
    └─ Legal Disclaimer Injection
    ↓
Structured Response + Sources + Confidence
```

## Quick Start (Local)

### 1. Install Dependencies
```bash
cd /Users/syedahmed/Enterprise\ Agentic\ RAG\ System/RAG_enterPriseSystem

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install packages
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env with your API keys:
# - GEMINI_API_KEY (for embeddings)
# - GROQ_API_KEY (for LLM)
# - QDRANT_CLUSTER_ENDPOINT + QDRANT_API_KEY
# - PORTKEY_API_KEY (for gateway)
# - LOGFIRE_TOKEN (optional, for observability)
```

### 3. Ingest Legal Corpus
```bash
# Ingest prototype legal corpus (statutes, treaties)
python -m app.ingestion.legal_loader --wipe

# Or ingest custom documents from DATA/
python -m app.ingestion.processor DATA --wipe
```

### 4. Start API Server
```bash
uvicorn app.main:app --reload --port 8000
```
API docs: http://localhost:8000/docs

### 5. Start Streamlit UI
```bash
cd ui
streamlit run app.py
```
UI: http://localhost:8501

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/query` | POST | Main RAG query (with guardrails, jurisdiction, citations) |
| `/classify` | POST | Quick formulation classification |
| `/jurisdictions` | GET | List supported jurisdictions |
| `/formulation-types` | GET | List formulation categories |
| `/graph` | GET | Mermaid graph PNG of agent workflow |

### Example Query
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"q": "Can I patent a classical Ayurvedic formulation?", "thread_id": "test-1"}'
```

### Example Classification
```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Herbal tablet with Ashwagandha and Guduchi for immunity",
    "ingredients": "Ashwagandha, Guduchi",
    "intended_use": "Therapeutic (treating disease)"
  }'
```

## Optional Fine-Tuning (RAFT, Phase 2)

The live MVP does not require fine-tuning. These scripts are kept as optional
Phase 2 scaffolding after the RAG pipeline and evaluation loop are stable.

### 1. Prepare Training Data
```bash
mkdir -p training_data
python -m app.finetuning.prepare_raft_dataset \
  --golden_dataset evals/golden_dataset_ayurveda_ip.json \
  --output training_data/raft_triples.jsonl
```

### 2. Train (GPU required, HF token needed)
```bash
# On GPU machine with LLaMA 3.1 8B access:
python -m app.finetuning.train_raft \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --data training_data/raft_triples.jsonl \
  --output_dir models/ip-sakti-llama3.1-8b-raft \
  --epochs 3
```

## Evaluation

```bash
cd evals
streamlit run app.py
```
Or programmatically:
```bash
python -m evals.pipeline
```

## Key Files

| File | Purpose |
|------|---------|
| `app/agents/graph.py` | LangGraph workflow with jurisdiction router, classifier, citation verifier |
| `app/agents/nodes/planner.py` | Intent classification (REGULATORY_IP, FORMULATION_CLASSIFICATION, CONVERSATIONAL, OFF_TOPIC) |
| `app/agents/nodes/jurisdiction_router.py` | Separates INDIA vs INTERNATIONAL law |
| `app/agents/nodes/formulation_classifier.py` | 5-category product classification |
| `app/agents/nodes/citation_verifier.py` | Claim extraction + evidence matching |
| `app/agents/nodes/responder.py` | Structured JSON output with disclaimer |
| `app/guardrails/colang_rules.py` | NeMo rules for scope, jailbreak, legal advice |
| `app/main.py` | FastAPI with /query, /classify, /jurisdictions |
| `ui/app.py` | Streamlit UI with jurisdiction toggle, classifier wizard, trust panel |
| `SOURCES_MANIFEST.md` | Authoritative source list with URLs and update cadence |
| `FINETUNING_CONFIG.md` | Optional RAFT fine-tuning guide for Phase 2 |
| `DATA/legal_corpus/` | Prototype legal text files with metadata headers |

## Prototype Status (≈70%)

✅ **Complete**
- Domain-adapted LangGraph agent (planner → router → classifier → retriever → responder → verifier)
- Jurisdiction separation (India / International / Both)
- Formulation classification (5 pathways + ABS flag)
- Source-cited responses with confidence
- NeMo guardrails (off-topic, jailbreak, legal advice)
- FastAPI with structured responses + /classify endpoint
- Streamlit UI with jurisdiction toggle, classifier wizard, "Why Trust This Answer" panel
- Prototype legal corpus (10+ statute/treaty files with metadata)
- Golden dataset (10 Ayurveda IP QA pairs + guardrails tests)
- Extended RAGAS metrics (faithfulness, citation correctness, jurisdiction accuracy, refusal accuracy, ABS accuracy)
- Optional RAFT fine-tuning scaffolding (dataset prep + training script)
- Dockerfile + requirements + env template

⚠️ **Needs Your Keys / Runtime**
- API keys in `.env` (Gemini, Groq, Qdrant, Portkey, Logfire)
- Vector DB population (run ingestion)
- Actual live LLM calls (requires Groq/Portkey connectivity; local grounded fallback is available for demos)

🔄 **Next for 100%**
- Full 500+ QA golden dataset
- Bhashini/IndicTrans2 multilingual integration
- Relational knowledge graph for multi-step reasoning
- Paid-source connectors (TKDL, paid IP databases) with consent logging
- Production hardening (DPDP audit, rate limits, auth)
- Voice interface

## License
Prototype for SIH 2026. Not for production legal use without qualified attorney review.
