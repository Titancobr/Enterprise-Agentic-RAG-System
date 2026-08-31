"""
Prepare RAFT-style fine-tuning dataset for IP-SAKTI Sahayak.

RAFT = Retrieval-Augmented Fine-Tuning:
- question
- retrieved authoritative contexts
- structured answer with citations

Usage:
python -m app.finetuning.prepare_raft_dataset \
  --golden_dataset evals/golden_dataset_ayurveda_ip.json \
  --output training_data/raft_triples.jsonl
"""

import argparse
import json
import os
from typing import Dict, Any, List


def build_instruction(sample: Dict[str, Any]) -> str:
    contexts = sample.get("relevant_contexts", [])
    context_text = "\n\n".join([f"SOURCE {i+1}: {ctx}" for i, ctx in enumerate(contexts)])
    
    return f"""You are IP-SAKTI Sahayak, a source-cited Ayurveda IP and regulatory assistant.
Use ONLY the retrieved contexts to answer. Keep India and international regimes separate.
Return structured JSON with answer, jurisdiction, formulation_type, claims, abs_required, and disclaimer.

RETRIEVED CONTEXTS:
{context_text}

QUESTION:
{sample['question']}"""


def build_response(sample: Dict[str, Any]) -> Dict[str, Any]:
    contexts = sample.get("relevant_contexts", [])
    claims = []
    
    for i, ctx in enumerate(contexts[:3]):
        source_hint = "Retrieved source"
        section_hint = ""
        if "Section" in ctx:
            section_hint = ctx.split(":")[0]
        elif "Article" in ctx:
            section_hint = ctx.split(":")[0]
        
        claims.append({
            "text": ctx[:180],
            "source": source_hint,
            "section": section_hint,
            "confidence": 0.9
        })
    
    return {
        "answer": sample.get("reference", ""),
        "jurisdiction": sample.get("expected_jurisdiction"),
        "formulation_type": sample.get("expected_formulation_type"),
        "claims": claims,
        "abs_required": sample.get("expected_abs_required"),
        "disclaimer": "This is general information, not legal advice. Consult a qualified IP attorney or regulatory expert for specific cases."
    }


def convert_to_chatml(sample: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": "You are IP-SAKTI Sahayak. Provide source-cited Ayurveda IP and regulatory guidance only."},
            {"role": "user", "content": build_instruction(sample)},
            {"role": "assistant", "content": json.dumps(build_response(sample), ensure_ascii=False)}
        ]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden_dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    with open(args.golden_dataset, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    count = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for sample in data.get("rag_samples", []):
            record = convert_to_chatml(sample)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    
    print(f"Wrote {count} RAFT training examples to {args.output}")


if __name__ == "__main__":
    main()
