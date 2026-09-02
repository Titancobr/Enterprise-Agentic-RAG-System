# ─────────────────────────────────────────────────────────────────────────────
# IP-SAKTI Sahayak — Ayurveda IP Test Runner
# Run: streamlit run evals/test_runner.py
# Requires: FastAPI backend running on localhost:8000
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys
import io
import json
import time
import copy
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from app.observability.logfire_compat import logfire
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"), service_name="test_runner")

import nest_asyncio
import asyncio
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
nest_asyncio.apply()

import requests
import pandas as pd
import streamlit as st

try:
    from audio_recorder_streamlit import audio_recorder
    AUDIO_RECORDER_AVAILABLE = True
except ImportError:
    AUDIO_RECORDER_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
DATASET_PATH = os.path.join(os.path.dirname(__file__), "golden_dataset_ayurveda_ip.json")
REQUEST_TIMEOUT = 120
DELAY_BETWEEN_CALLS = 8   # seconds — stay within Groq RPM

DOMAIN_COLORS = {
    "patent_eligibility":        "🔵",
    "abs_compliance":            "🟢",
    "formulation_classification":"🟠",
    "gi_protection":             "🟣",
    "food_vs_drug":              "🟡",
    "phytopharmaceutical":       "🔴",
    "cosmetic":                  "🩷",
    "international":             "🌍",
    "edge_case":                 "⚫",
}

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IP-SAKTI — Test Runner",
    page_icon="🧪",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_dataset() -> dict:
    with open(DATASET_PATH) as f:
        return json.load(f)


def _keyword_score(response: str, reference: str) -> float:
    """Overlap of significant reference keywords in the actual response (case-insensitive)."""
    if not response or not reference:
        return 0.0
    ref_words = set(w.lower() for w in reference.split() if len(w) > 4)
    if not ref_words:
        return 0.0
    hits = sum(1 for w in ref_words if w in response.lower())
    return round(hits / len(ref_words), 3)


def _formulation_match(actual: str | None, expected: str | None) -> bool:
    if expected is None:
        return True   # not applicable for this sample
    if actual is None:
        return False
    return actual.upper() == expected.upper()


def _abs_match(actual, expected) -> bool:
    if expected is None:
        return True   # not applicable
    if actual is None:
        return False
    # API returns bool or string
    if isinstance(actual, str):
        actual_bool = actual.lower() in ("true", "yes", "1")
    else:
        actual_bool = bool(actual)
    return actual_bool == bool(expected)


def _badge(score: float) -> str:
    if score >= 0.75:
        return "🟢"
    if score >= 0.50:
        return "🟡"
    return "🔴"


def _grade(score: float) -> str:
    if score >= 0.75:
        return "✅ Good"
    if score >= 0.50:
        return "⚠️ Fair"
    return "❌ Poor"


def _transcribe_audio_groq(audio_bytes: bytes) -> str:
    try:
        from groq import Groq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            st.error("GROQ_API_KEY not set in .env")
            return ""
        client = Groq(api_key=api_key)
        buf = io.BytesIO(audio_bytes)
        buf.name = "voice_input.wav"
        transcription = client.audio.transcriptions.create(
            file=buf,
            model="whisper-large-v3-turbo",
            response_format="text",
            language="en",
        )
        return transcription.strip() if isinstance(transcription, str) else transcription.text.strip()
    except Exception as e:
        st.error(f"Whisper transcription failed: {e}")
        return ""


def _call_backend(question: str, jurisdiction: str = "INDIA") -> dict:
    try:
        r = requests.post(
            f"{BACKEND_URL}/query",
            json={"q": question, "thread_id": f"test_runner_{int(time.time())}", "jurisdiction": jurisdiction},
            timeout=REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Backend offline — start FastAPI on :8000"}
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────────────────────────────────────────
if "dataset" not in st.session_state:
    st.session_state.dataset = _load_dataset()
if "test_results" not in st.session_state:
    st.session_state.test_results = []
if "phase2_results" not in st.session_state:
    st.session_state.phase2_results = {}
if "run_done" not in st.session_state:
    st.session_state.run_done = False
if "voice_test_transcript" not in st.session_state:
    st.session_state.voice_test_transcript = ""
if "last_voice_bytes" not in st.session_state:
    st.session_state.last_voice_bytes = None
if "selected_domains" not in st.session_state:
    st.session_state.selected_domains = []

dataset = st.session_state.dataset
samples = dataset["rag_samples"]
all_domains = sorted(set(s.get("domain", "unknown") for s in samples))

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.title("🧪 IP-SAKTI — Ayurveda IP Test Runner")
st.caption(
    "Tests the full RAG pipeline on domain-specific Ayurveda IP questions. "
    "Phase 1 collects live responses · Phase 2 scores with RAGAS metrics."
)
st.warning(
    "⚠️ Make sure the FastAPI backend is running: `uvicorn app.main:app --reload --port 8000`",
    icon="⚠️",
)
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — filter controls
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧪 Test Controls")
    st.markdown("### Filter by Domain")
    st.session_state.selected_domains = st.multiselect(
        "Domains to test (empty = all):",
        options=all_domains,
        default=[],
        format_func=lambda d: f"{DOMAIN_COLORS.get(d, '⚪')} {d.replace('_', ' ').title()}",
    )
    st.markdown("---")
    st.markdown("### Dataset")
    st.metric("Total Questions", len(samples))
    domain_counts = {}
    for s in samples:
        d = s.get("domain", "unknown")
        domain_counts[d] = domain_counts.get(d, 0) + 1
    for domain, count in sorted(domain_counts.items()):
        icon = DOMAIN_COLORS.get(domain, "⚪")
        st.caption(f"{icon} {domain.replace('_', ' ').title()}: **{count}**")
    st.markdown("---")
    if st.button("🗑️ Reset Results", type="secondary", use_container_width=True):
        st.session_state.test_results = []
        st.session_state.phase2_results = {}
        st.session_state.run_done = False
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_questions, tab_run, tab_accuracy, tab_ragas, tab_voice = st.tabs([
    "📋 Question Bank",
    "▶️ Run Tests",
    "📊 Accuracy Dashboard",
    "🧪 RAGAS Metrics",
    "🎙️ Voice Test",
])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — Question Bank
# ═══════════════════════════════════════════════════════════════════════════
with tab_questions:
    st.subheader("Ayurveda IP Golden Dataset")
    st.markdown(
        f"**{len(samples)} questions** across **{len(all_domains)} domains**. "
        "Each has a reference answer, expected formulation type, ABS flag, and jurisdiction."
    )

    filter_domains = st.session_state.selected_domains or all_domains
    rows = []
    for s in samples:
        if s.get("domain", "unknown") not in filter_domains:
            continue
        rows.append({
            "ID": s["id"],
            "Domain": f"{DOMAIN_COLORS.get(s.get('domain',''), '⚪')} {s.get('domain','').replace('_',' ').title()}",
            "Question": s["question"],
            "Expected Pathway": s.get("expected_formulation_type") or "—",
            "ABS": "✅" if s.get("expected_abs_required") else ("❌" if s.get("expected_abs_required") is False else "?"),
            "Jurisdiction": s.get("expected_jurisdiction", "INDIA"),
            "Reference (preview)": (s.get("reference", "")[:100] + "…") if len(s.get("reference", "")) > 100 else s.get("reference", ""),
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with st.expander("📄 View Raw JSON Dataset"):
        st.json(dataset)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — Run Tests
# ═══════════════════════════════════════════════════════════════════════════
with tab_run:
    st.subheader("Phase 1 — Live Pipeline Test")

    filter_domains_run = st.session_state.selected_domains or all_domains
    filtered_samples = [s for s in samples if s.get("domain", "unknown") in filter_domains_run]
    st.info(f"Running **{len(filtered_samples)}** question(s) from **{len(filter_domains_run)}** domain(s). "
            f"Estimated time: ~{len(filtered_samples) * (DELAY_BETWEEN_CALLS + 15) // 60 + 1} min.")

    col1, col2 = st.columns([1, 3])
    run_btn = col1.button("▶️ Run All Tests", type="primary", use_container_width=True, disabled=st.session_state.run_done)
    col2.caption("Each question is sent to the live backend. Results are scored automatically.")

    if run_btn:
        progress_bar = st.progress(0, text="Starting…")
        result_placeholder = st.empty()
        results = []

        for i, sample in enumerate(filtered_samples):
            q = sample["question"]
            jurisdiction = sample.get("expected_jurisdiction", "INDIA")
            progress_bar.progress((i) / len(filtered_samples), text=f"[{i+1}/{len(filtered_samples)}] {q[:60]}…")

            with st.status(f"🔍 Q{sample['id']}: {q[:60]}…", expanded=False) as status:
                data = _call_backend(q, jurisdiction)

                if "error" in data:
                    status.update(label=f"❌ Q{sample['id']}: Backend error", state="error")
                    results.append({
                        "id": sample["id"],
                        "domain": sample.get("domain", ""),
                        "question": q,
                        "reference": sample.get("reference", ""),
                        "actual_response": "",
                        "keyword_score": 0.0,
                        "formulation_correct": False,
                        "abs_correct": False,
                        "confidence_label": None,
                        "expected_formulation_type": sample.get("expected_formulation_type"),
                        "actual_formulation_type": None,
                        "expected_abs": sample.get("expected_abs_required"),
                        "actual_abs": None,
                        "sources_count": 0,
                        "error": data["error"],
                        "actual_contexts": [],
                        "actual_tools_called": [],
                    })
                    if i < len(filtered_samples) - 1:
                        time.sleep(DELAY_BETWEEN_CALLS)
                    continue

                actual_response = data.get("answer") or ""
                actual_ft = data.get("formulation_type")
                actual_abs = data.get("abs_required")
                confidence_label = data.get("confidence_label")
                sources = data.get("sources") or []
                thought = data.get("thought_process") or []

                k_score = _keyword_score(actual_response, sample.get("reference", ""))
                ft_correct = _formulation_match(actual_ft, sample.get("expected_formulation_type"))
                abs_correct = _abs_match(actual_abs, sample.get("expected_abs_required"))
                overall = "✅ Pass" if k_score >= 0.4 and ft_correct and abs_correct else "⚠️ Partial" if k_score >= 0.2 else "❌ Fail"

                status.update(label=f"{overall} Q{sample['id']}: {q[:50]}…", state="complete" if k_score >= 0.4 else "error")
                st.markdown(f"**Answer preview:** {actual_response[:200]}…" if len(actual_response) > 200 else f"**Answer:** {actual_response}")
                st.caption(f"Keyword score: `{k_score:.3f}` | Formulation: {'✅' if ft_correct else '❌'} | ABS: {'✅' if abs_correct else '❌'} | Confidence: `{confidence_label or 'N/A'}`")

                results.append({
                    "id": sample["id"],
                    "domain": sample.get("domain", ""),
                    "question": q,
                    "reference": sample.get("reference", ""),
                    "actual_response": actual_response,
                    "keyword_score": k_score,
                    "formulation_correct": ft_correct,
                    "abs_correct": abs_correct,
                    "confidence_label": confidence_label,
                    "expected_formulation_type": sample.get("expected_formulation_type"),
                    "actual_formulation_type": actual_ft,
                    "expected_abs": sample.get("expected_abs_required"),
                    "actual_abs": actual_abs,
                    "sources_count": len(sources),
                    "error": None,
                    "actual_contexts": sources[:5],
                    "actual_tools_called": ["retrieve_documents"] if sources else ["direct_answer"],
                })

            if i < len(filtered_samples) - 1:
                time.sleep(DELAY_BETWEEN_CALLS)

        progress_bar.progress(1.0, text="✅ All tests complete!")
        st.session_state.test_results = results
        st.session_state.run_done = True
        st.success(f"✅ Phase 1 complete — {len(results)} questions tested.")
        st.rerun()

    # Show last results summary if already run
    if st.session_state.run_done and st.session_state.test_results:
        results = st.session_state.test_results
        df = pd.DataFrame(results)

        st.markdown("### Phase 1 Results Summary")
        col_a, col_b, col_c, col_d = st.columns(4)
        avg_kw = df["keyword_score"].mean()
        ft_acc = df["formulation_correct"].mean()
        abs_acc = df["abs_correct"].mean()
        pass_rate = (df["keyword_score"] >= 0.4).mean()
        col_a.metric("Avg Keyword Score", f"{_badge(avg_kw)} {avg_kw:.2f}")
        col_b.metric("Formulation Accuracy", f"{_badge(ft_acc)} {ft_acc:.0%}")
        col_c.metric("ABS Flag Accuracy", f"{_badge(abs_acc)} {abs_acc:.0%}")
        col_d.metric("Pass Rate (≥0.4 kw)", f"{_badge(pass_rate)} {pass_rate:.0%}")

        display_cols = ["id", "domain", "question", "keyword_score", "formulation_correct", "abs_correct", "confidence_label", "sources_count"]
        df_show = df[display_cols].copy()
        df_show["domain"] = df_show["domain"].apply(lambda d: f"{DOMAIN_COLORS.get(d,'⚪')} {d.replace('_',' ').title()}")

        def _color_kw(val):
            if not isinstance(val, float):
                return ""
            if val >= 0.75:
                return "background-color: #d4edda"
            if val >= 0.4:
                return "background-color: #fff3cd"
            return "background-color: #f8d7da"

        styled = df_show.style.applymap(_color_kw, subset=["keyword_score"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # Download
        csv = df.to_csv(index=False)
        st.download_button("📥 Download Full Results CSV", csv, "test_results.csv", "text/csv")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — Accuracy Dashboard
# ═══════════════════════════════════════════════════════════════════════════
with tab_accuracy:
    st.subheader("Accuracy Dashboard")

    if not st.session_state.run_done or not st.session_state.test_results:
        st.info("Run Phase 1 first (▶️ Run Tests tab) to see accuracy results.")
    else:
        results = st.session_state.test_results
        df = pd.DataFrame(results)

        # ── Overall metrics
        st.markdown("### Overall Metrics")
        cols = st.columns(4)
        avg_kw = df["keyword_score"].mean()
        ft_acc = df[df["expected_formulation_type"].notna()]["formulation_correct"].mean() if df["expected_formulation_type"].notna().any() else 0.0
        abs_acc = df[df["expected_abs"].notna()]["abs_correct"].mean() if df["expected_abs"].notna().any() else 0.0
        pass_rate = (df["keyword_score"] >= 0.4).mean()
        cols[0].metric("Keyword Overlap", f"{avg_kw:.3f}", help="Fraction of reference keywords found in response")
        cols[1].metric("Formulation Accuracy", f"{ft_acc:.0%}", help="Did the pipeline classify the pathway correctly?")
        cols[2].metric("ABS Flag Accuracy", f"{abs_acc:.0%}", help="Did the pipeline flag ABS correctly?")
        cols[3].metric("Pass Rate", f"{pass_rate:.0%}", help="Fraction of questions with keyword score ≥ 0.4")

        st.divider()

        # ── Per-domain breakdown
        st.markdown("### Per-Domain Accuracy")
        domain_rows = []
        for domain in sorted(df["domain"].unique()):
            sub = df[df["domain"] == domain]
            sub_ft = sub[sub["expected_formulation_type"].notna()]
            sub_abs = sub[sub["expected_abs"].notna()]
            domain_rows.append({
                "Domain": f"{DOMAIN_COLORS.get(domain,'⚪')} {domain.replace('_',' ').title()}",
                "Questions": len(sub),
                "Avg Keyword Score": round(sub["keyword_score"].mean(), 3),
                "Pass Rate (≥0.4)": f"{(sub['keyword_score'] >= 0.4).mean():.0%}",
                "Formulation Acc.": f"{sub_ft['formulation_correct'].mean():.0%}" if len(sub_ft) > 0 else "N/A",
                "ABS Acc.": f"{sub_abs['abs_correct'].mean():.0%}" if len(sub_abs) > 0 else "N/A",
            })
        st.dataframe(pd.DataFrame(domain_rows), use_container_width=True, hide_index=True)

        st.divider()

        # ── Weak spots
        st.markdown("### ⚠️ Weak Spots (keyword score < 0.4)")
        weak = df[df["keyword_score"] < 0.4][["id", "domain", "question", "keyword_score", "formulation_correct", "abs_correct", "error"]]
        if len(weak) == 0:
            st.success("🎉 No weak spots — all questions scored ≥ 0.4!")
        else:
            st.warning(f"{len(weak)} question(s) scored below threshold.")
            st.dataframe(weak, use_container_width=True, hide_index=True)

        st.divider()

        # ── Confidence calibration
        st.markdown("### Confidence Calibration")
        conf_rows = []
        for _, row in df.iterrows():
            conf_rows.append({
                "Question": row["question"][:60] + "…",
                "Confidence Label": row["confidence_label"] or "N/A",
                "Keyword Score": row["keyword_score"],
                "Calibrated?": "✅" if (
                    (row["confidence_label"] == "High" and row["keyword_score"] >= 0.65) or
                    (row["confidence_label"] == "Medium" and 0.35 <= row["keyword_score"] < 0.65) or
                    (row["confidence_label"] == "Low" and row["keyword_score"] < 0.35) or
                    row["confidence_label"] is None
                ) else "❌"
            })
        st.dataframe(pd.DataFrame(conf_rows), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — RAGAS Metrics (Phase 2)
# ═══════════════════════════════════════════════════════════════════════════
with tab_ragas:
    st.subheader("Phase 2 — RAGAS Evaluation")
    st.markdown(
        "Runs **Faithfulness, Answer Relevancy, Context Precision, Context Recall, Answer Correctness, Tool Correctness** "
        "using the Groq judge LLM. This takes ~20 minutes due to Groq rate limits."
    )

    if not st.session_state.run_done or not st.session_state.test_results:
        st.info("Complete Phase 1 first (▶️ Run Tests tab).")
    else:
        ragas_btn = st.button(
            "🚀 Run RAGAS Phase 2",
            type="primary",
            disabled=bool(st.session_state.phase2_results),
        )
        if st.session_state.phase2_results:
            st.info("RAGAS results already computed. Reset from sidebar to re-run.")

        if ragas_btn:
            # Build a golden-dataset-compatible dict from our Phase 1 results
            phase1_results = st.session_state.test_results
            ragas_samples = []
            for r in phase1_results:
                if not r.get("actual_response"):
                    continue
                ragas_samples.append({
                    "question": r["question"],
                    "reference": r["reference"],
                    "actual_response": r["actual_response"],
                    "actual_contexts": r.get("actual_contexts", []),
                    "actual_tools_called": r.get("actual_tools_called", []),
                    "expected_tools": ["retrieve_documents"],
                })
            ragas_dataset = {"rag_samples": ragas_samples}

            status_box = st.empty()

            def _status_cb(msg: str):
                status_box.info(msg)

            from evals.metrics import run_all_metrics

            async def _run():
                return await run_all_metrics(ragas_dataset, status_cb=_status_cb)

            loop = asyncio.get_event_loop()
            with st.spinner("🧪 Running RAGAS experiments… (this will take ~20 minutes)"):
                try:
                    metric_results = loop.run_until_complete(_run())
                    st.session_state.phase2_results = metric_results
                    st.success("✅ RAGAS Phase 2 complete!")
                    st.rerun()
                except Exception as e:
                    st.error(f"RAGAS failed: {e}")

        if st.session_state.phase2_results:
            st.markdown("### RAGAS Results")
            SCORE_COLORS = {"green": "#d4edda", "yellow": "#fff3cd", "red": "#f8d7da"}

            def _color_score(val):
                if not isinstance(val, (int, float)):
                    return ""
                if val >= 0.75:
                    return f"background-color: {SCORE_COLORS['green']}"
                if val >= 0.5:
                    return f"background-color: {SCORE_COLORS['yellow']}"
                return f"background-color: {SCORE_COLORS['red']}"

            for metric_name, df_metric in st.session_state.phase2_results.items():
                avg = df_metric[metric_name].mean()
                label = "✅ Good" if avg >= 0.75 else "⚠️ Fair" if avg >= 0.5 else "❌ Poor"
                st.markdown(f"**{metric_name.replace('_', ' ').title()}** — AVG: {_badge(avg)} `{avg:.2f}` {label}")
                styled = df_metric.style.applymap(_color_score, subset=[metric_name]).format({metric_name: "{:.3f}"})
                st.dataframe(styled, use_container_width=True, hide_index=True)
                st.divider()

            # Download all as CSV
            all_dfs = []
            for metric_name, df_metric in st.session_state.phase2_results.items():
                df_copy = df_metric.copy()
                df_copy["metric"] = metric_name
                all_dfs.append(df_copy)
            if all_dfs:
                combined = pd.concat(all_dfs, ignore_index=True)
                st.download_button("📥 Download RAGAS CSV", combined.to_csv(index=False), "ragas_results.csv", "text/csv")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 5 — Voice Test
# ═══════════════════════════════════════════════════════════════════════════
with tab_voice:
    st.subheader("🎙️ Voice Query Test")
    st.markdown(
        "Speak or type a question. The system will transcribe via **Groq Whisper**, "
        "send it to the backend, and score the response against a reference answer you provide."
    )

    col_v1, col_v2 = st.columns([1, 2])

    with col_v1:
        st.markdown("#### 1. Record or Type Query")
        if AUDIO_RECORDER_AVAILABLE:
            st.caption("Click the mic to record:")
            voice_bytes = audio_recorder(
                text="",
                recording_color="#e74c3c",
                neutral_color="#27ae60",
                icon_name="microphone",
                icon_size="2x",
                pause_threshold=2.5,
                key="voice_test_recorder",
            )
            if voice_bytes and voice_bytes != st.session_state.last_voice_bytes:
                st.session_state.last_voice_bytes = voice_bytes
                with st.spinner("🎧 Transcribing…"):
                    transcript = _transcribe_audio_groq(voice_bytes)
                if transcript:
                    st.session_state.voice_test_transcript = transcript
                    st.success(f"✅ Transcript: *{transcript}*")
                else:
                    st.warning("Could not transcribe. Try again.")
        else:
            st.warning("`audio-recorder-streamlit` not available.")

        typed = st.text_area(
            "Or type your query:",
            value=st.session_state.voice_test_transcript,
            height=100,
            key="voice_typed_query",
        )
        jurisdiction_v = st.selectbox("Jurisdiction:", ["INDIA", "INTERNATIONAL", "BOTH"], key="voice_jurisdiction")
        reference_v = st.text_area(
            "Reference answer (optional — for scoring):",
            placeholder="Paste expected answer to compute keyword score…",
            height=80,
        )

        send_btn = st.button("🚀 Send to Backend", type="primary", use_container_width=True)

    with col_v2:
        st.markdown("#### 2. Live Response")
        if send_btn:
            query = typed.strip()
            if not query:
                st.warning("Please provide a query (type or record).")
            else:
                with st.spinner(f"🔍 Querying backend for: *{query[:60]}…*"):
                    data = _call_backend(query, jurisdiction_v)

                if "error" in data:
                    st.error(f"Backend error: {data['error']}")
                else:
                    answer = data.get("answer") or "No answer returned."
                    intent = data.get("intent", "")
                    confidence = data.get("confidence_label", "N/A")
                    formulation = data.get("formulation_type", "N/A")
                    abs_req = data.get("abs_required")
                    sources = data.get("sources") or []

                    st.markdown("**Answer:**")
                    st.markdown(answer)
                    st.divider()

                    mc1, mc2, mc3, mc4 = st.columns(4)
                    mc1.metric("Intent", intent or "N/A")
                    mc2.metric("Confidence", confidence)
                    mc3.metric("Formulation", formulation or "N/A")
                    mc4.metric("ABS Required", "Yes" if abs_req else ("No" if abs_req is False else "?"))

                    if reference_v.strip():
                        kw = _keyword_score(answer, reference_v.strip())
                        st.markdown(f"**Keyword Score vs Reference:** {_badge(kw)} `{kw:.3f}` — {_grade(kw)}")

                    if sources:
                        with st.expander(f"📄 Retrieved Sources ({len(sources)})"):
                            for i, src in enumerate(sources):
                                st.info(f"**Source {i+1}:** {src[:200]}…" if len(src) > 200 else src)

        else:
            st.info("Record or type a question, then click **Send to Backend**.")

        # ── Sample questions for quick testing
        st.markdown("#### 3. Quick Test Questions")
        st.caption("Click any question to pre-fill the query box:")
        sample_qs = [
            "Can I patent a classical Ayurvedic formulation mentioned in Charaka Samhita?",
            "My startup wants to export Ashwagandha products. Do I need ABS approval?",
            "I have a new herbal tablet combining Ashwagandha, Guduchi, Brahmi — how should I classify it?",
            "Can I register a Geographical Indication for Kerala Ayurvedic oil?",
            "I want to sell Ashwagandha churna as a health supplement. What regulations apply?",
            "Is a phytopharmaceutical the same as a classical Ayurvedic drug?",
            "What is the Nagoya Protocol and does it apply to my Ayurveda business?",
        ]
        for sq in sample_qs:
            if st.button(f"📌 {sq[:65]}…" if len(sq) > 65 else f"📌 {sq}", use_container_width=True):
                st.session_state.voice_test_transcript = sq
                st.rerun()
