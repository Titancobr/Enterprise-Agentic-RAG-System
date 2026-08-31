# IP-SAKTI Sahayak — Fine-Tuning Configuration

## Recommended Approach: Open-Weight LLaMA 3.1 8B (RAFT-style)

Per the problem statement requirements and the discussion, we use **open-weight models** (not proprietary APIs) for fine-tuning with the RAFT methodology.

### Base Model
- **Model**: LLaMA 3.1 8B Instruct (Meta)
- **License**: LLaMA Community License (commercial use permitted)
- **Weights**: Available on Hugging Face (`meta-llama/Llama-3.1-8B-Instruct`)

### Why RAFT (Retrieval-Augmented Fine-Tuning)?
- Model learns to answer questions using retrieved context, not just pretrained knowledge
- Enforces citation grounding
- Produces structured JSON output with jurisdiction, confidence, citations
- Adaptable to legal/regulatory domain vocabulary

---

## Training Data Format (RAFT Triples)

Each training example contains:

```json
{
  "question": "Can I patent a classical Ayurvedic formulation?",
  "retrieved_contexts": [
    "[Patents Act 1970, Section 3(p)] An invention which, in effect, is traditional knowledge or which is an aggregation or duplication of known properties of traditionally known component or components.",
    "[Drugs & Cosmetics Act, First Schedule] Formulations whose composition and method of preparation are EXACTLY as described in authoritative texts are Classical Ayurvedic Medicines."
  ],
  "answer": {
    "text": "Classical Ayurvedic formulations documented in the First Schedule of the Drugs & Cosmetics Act cannot be patented as products due to Section 3(p) of the Patents Act, which bars patenting traditional knowledge. However, novel manufacturing processes may be patentable if they meet the standard criteria.",
    "jurisdiction": "INDIA",
    "formulation_type": "CLASSICAL_AYURVEDIC",
    "claims": [
      {"text": "Section 3(p) bars patenting traditional knowledge", "source": "Patents Act 1970", "section": "3(p)", "confidence": 0.95},
      {"text": "Classical formulations are in First Schedule", "source": "Drugs & Cosmetics Act", "section": "First Schedule", "confidence": 0.92}
    ],
    "abs_required": false,
    "disclaimer": "This is general information, not legal advice. Consult a qualified IP attorney for specific cases."
  }
}
```

---

## Fine-Tuning Pipeline

### 1. Prepare Training Data
```bash
# Generate RAFT triples from golden dataset + retrieved contexts
python -m app.finetuning.prepare_raft_dataset \
  --golden_dataset evals/golden_dataset_ayurveda_ip.json \
  --output training_data/raft_triples.jsonl \
  --context-limit 3
```

### 2. LoRA Configuration (Parameter-Efficient Fine-Tuning)
```yaml
# lora_config.yaml
model: "meta-llama/Llama-3.1-8B-Instruct"
method: "lora"
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
```

### 3. Training Script (using Hugging Face TRL)
```bash
# Using Unsloth for fast training on consumer GPU
pip install unsloth trl peft bitsandbytes

python -m app.finetuning.train_raft \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --data training_data/raft_triples.jsonl \
  --output_dir models/ip-sakti-llama3.1-8b-raft \
  --epochs 3 \
  --batch_size 4 \
  --learning_rate 2e-4 \
  --max_seq_length 4096
```

### 4. Inference with Fine-Tuned Model
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = PeftModel.from_pretrained(base_model, "models/ip-sakti-llama3.1-8b-raft")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
```

---

## Model Outputs

The fine-tuned model is trained to produce structured JSON:

```json
{
  "answer": "...",
  "jurisdiction": "INDIA" | "INTERNATIONAL" | "BOTH",
  "formulation_type": "CLASSICAL_AYURVEDIC" | "PROPRIETARY_AYURVEDIC" | "PHYTOPHARMACEUTICAL" | "FOOD_AYURVEDA_AAHAR" | "COSMETIC" | "INSUFFICIENT_INFO",
  "claims": [
    {"text": "...", "source": "...", "section": "...", "confidence": 0.0-1.0}
  ],
  "abs_required": true | false | null,
  "disclaimer": "This is general information, not legal advice."
}
```

---

## Evaluation Metrics (Post Fine-Tuning)

| Metric | Target | Measurement |
|--------|--------|-------------|
| Faithfulness | > 0.90 | RAGAS |
| Citation Correctness | > 0.85 | Custom |
| Jurisdiction Accuracy | > 0.95 | Custom |
| Refusal Accuracy | > 0.90 | Custom |
| Hallucination Rate | < 0.05 | FActScore |

---

## Deployment Options

1. **Local GPU**: Run inference on-premises (8GB+ VRAM for 8B model with 4-bit quantization)
2. **Groq Cloud**: Deploy quantized model via Groq API for low-latency inference
3. **vLLM**: High-throughput serving with PagedAttention

---

## Data Sources for Fine-Tuning (Per Problem Statement)

The corpus for fine-tuning should be assembled from:

- **TKDL** (tkdl.res.in) — Traditional Knowledge Digital Library
- **India Code** (indiacode.nic.in) — Statutes & rules
- **IP India** (ipindia.gov.in) — Patent, TM, GI databases
- **NBA** (nbaindia.org) — Biodiversity Authority
- **FSSAI** — Ayurveda Aahar regulations
- **WIPO** — GRATK Treaty, PCT, Madrid, Hague

> ⚠️ Only use publicly available, non-paywalled content for training without explicit user consent.
