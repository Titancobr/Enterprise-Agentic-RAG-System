#!/usr/bin/env python3
"""
CI/CD Evaluation Gate for IP-SAKTI Sahayak
Runs all benchmark questions in golden_dataset_ayurveda_ip.json against the backend
and enforces production performance thresholds:
- Overall Pass Rate: >= 85%
- ABS Accuracy: >= 90%
- Average Keyword Overlap: >= 0.600
"""

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
evals_dir = os.path.abspath(os.path.dirname(__file__))
if evals_dir in sys.path:
    sys.path.remove(evals_dir)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import json
import time
import requests

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
DATASET_PATH = os.path.join(os.path.dirname(__file__), "golden_dataset_ayurveda_ip.json")
DELAY_BETWEEN_CALLS = float(os.getenv("DELAY_BETWEEN_CALLS", "1.0"))

THRESHOLD_PASS_RATE = 0.85
THRESHOLD_ABS_ACCURACY = 0.90
THRESHOLD_KEYWORD_OVERLAP = 0.600


def keyword_score(response: str, reference: str) -> float:
    if not response or not reference:
        return 0.0
    ref_words = set(w.lower() for w in reference.split() if len(w) > 4)
    if not ref_words:
        return 0.0
    hits = sum(1 for w in ref_words if w in response.lower())
    return round(hits / len(ref_words), 3)


def formulation_match(actual: str | None, expected: str | None) -> bool:
    if expected is None:
        return True
    if actual is None:
        return False
    return actual.upper() == expected.upper()


def abs_match(actual, expected) -> bool:
    if expected is None:
        return True
    if actual is None:
        return False
    if isinstance(actual, str):
        actual_bool = actual.lower() in ("true", "yes", "1")
    else:
        actual_bool = bool(actual)
    return actual_bool == bool(expected)


def run_eval(sample_limit: int | None = None):
    print("=" * 70)
    print("🚀 IP-SAKTI Sahayak — Production CI/CD Evaluation Gate")
    print(f"📡 Backend URL: {BACKEND_URL}")
    print(f"📂 Dataset: {DATASET_PATH}")
    print("=" * 70)

    client = None
    use_in_process = False

    # Check backend health
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=3)
        if r.status_code == 200:
            print("✅ Connected to live backend at", BACKEND_URL)
        else:
            raise RuntimeError(f"HTTP {r.status_code}")
    except Exception:
        print(f"ℹ️ Live backend not reachable at {BACKEND_URL}. Using in-process FastAPI TestClient...")
        from fastapi.testclient import TestClient
        from app.main import app
        client = TestClient(app)
        use_in_process = True

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = data.get("rag_samples", [])
    if sample_limit:
        samples = samples[:sample_limit]

    print(f"🧪 Evaluating {len(samples)} benchmark questions...\n")

    results = []
    for i, sample in enumerate(samples, 1):
        qid = sample["id"]
        q = sample["question"]
        ref = sample.get("reference", "")
        exp_ft = sample.get("expected_formulation_type")
        exp_abs = sample.get("expected_abs_required")
        jurisdiction = sample.get("expected_jurisdiction", "INDIA")

        payload = {
            "q": q,
            "thread_id": f"ci_eval_{int(time.time())}_{qid}",
            "jurisdiction": jurisdiction
        }

        try:
            if use_in_process:
                resp = client.post("/query", json=payload)
            else:
                resp = requests.post(f"{BACKEND_URL}/query", json=payload, timeout=120)
            resp.raise_for_status()
            res_data = resp.json()
            answer = res_data.get("answer", "")
            act_ft = res_data.get("formulation_type")
            act_abs = res_data.get("abs_required")
        except Exception as e:
            print(f"[{i}/{len(samples)}] Q{qid}: ERROR - {e}")
            results.append({
                "id": qid,
                "domain": sample.get("domain", ""),
                "kw_score": 0.0,
                "ft_ok": False,
                "abs_ok": False,
                "passed": False,
                "error": str(e)
            })
            time.sleep(DELAY_BETWEEN_CALLS)
            continue

        kw = keyword_score(answer, ref)
        ft_ok = formulation_match(act_ft, exp_ft)
        abs_ok = abs_match(act_abs, exp_abs)
        passed = (kw >= 0.40) and ft_ok and abs_ok

        status_emoji = "✅" if passed else "⚠️" if kw >= 0.30 else "❌"
        print(f"[{i:02d}/{len(samples):02d}] {status_emoji} Q{qid:<2} | KW: {kw:.3f} | FT: {'OK' if ft_ok else 'FAIL'} | ABS: {'OK' if abs_ok else 'FAIL'} | {q[:55]}...")

        results.append({
            "id": qid,
            "domain": sample.get("domain", ""),
            "kw_score": kw,
            "ft_ok": ft_ok,
            "abs_ok": abs_ok,
            "passed": passed,
            "error": None
        })

        if i < len(samples):
            time.sleep(DELAY_BETWEEN_CALLS)

    # Aggregated metrics
    total = len(results)
    avg_kw = sum(r["kw_score"] for r in results) / total if total else 0.0
    abs_correct_count = sum(1 for r in results if r["abs_ok"])
    abs_acc = abs_correct_count / total if total else 0.0
    ft_correct_count = sum(1 for r in results if r["ft_ok"])
    ft_acc = ft_correct_count / total if total else 0.0
    pass_count = sum(1 for r in results if r["passed"])
    pass_rate = pass_count / total if total else 0.0

    print("\n" + "=" * 70)
    print("📊 EVALUATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"Total Questions Evaluated:  {total}")
    print(f"Overall Pass Rate:          {pass_rate:.1%}  (Threshold: {THRESHOLD_PASS_RATE:.0%}) -> {'PASSED ✅' if pass_rate >= THRESHOLD_PASS_RATE else 'FAILED ❌'}")
    print(f"ABS Accuracy:               {abs_acc:.1%}  (Threshold: {THRESHOLD_ABS_ACCURACY:.0%}) -> {'PASSED ✅' if abs_acc >= THRESHOLD_ABS_ACCURACY else 'FAILED ❌'}")
    print(f"Formulation Accuracy:       {ft_acc:.1%}")
    print(f"Average Keyword Overlap:    {avg_kw:.3f}  (Threshold: {THRESHOLD_KEYWORD_OVERLAP:.3f}) -> {'PASSED ✅' if avg_kw >= THRESHOLD_KEYWORD_OVERLAP else 'FAILED ❌'}")
    print("=" * 70)

    all_passed = (pass_rate >= THRESHOLD_PASS_RATE) and (abs_acc >= THRESHOLD_ABS_ACCURACY) and (avg_kw >= THRESHOLD_KEYWORD_OVERLAP)
    if all_passed:
        print("\n🎉 ALL CI/CD PRODUCTION GATES PASSED! Ready for deployment.\n")
        sys.exit(0)
    else:
        print("\n❌ CI/CD GATES NOT MET. Please review the failed questions above.\n")
        sys.exit(1)


if __name__ == "__main__":
    limit = None
    if len(sys.argv) > 1:
        if sys.argv[1] in ("--limit", "-l") and len(sys.argv) > 2:
            limit = int(sys.argv[2])
        else:
            try:
                limit = int(sys.argv[1])
            except ValueError:
                limit = None
    run_eval(sample_limit=limit)
