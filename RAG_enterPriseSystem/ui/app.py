import os
import streamlit as st
import requests
import time
import uuid
from dotenv import load_dotenv

try:
    import logfire
except ModuleNotFoundError:
    import logging
    from contextlib import contextmanager

    logging.basicConfig(level=logging.INFO)
    _logger = logging.getLogger("ip_sakti_ui")

    @contextmanager
    def _noop_span(*args, **kwargs):
        yield None

    class _LogfireFallback:
        def configure(self, *args, **kwargs):
            return None

        def span(self, *args, **kwargs):
            return _noop_span(*args, **kwargs)

        def info(self, message, *args, **kwargs):
            _logger.info(message)

        def warning(self, message, *args, **kwargs):
            _logger.warning(message)

        def warn(self, message, *args, **kwargs):
            self.warning(message)

        def error(self, message, *args, **kwargs):
            _logger.error(message)

    logfire = _LogfireFallback()


env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv(dotenv_path=env_path)


try:
    token = os.getenv("LOGFIRE_TOKEN")
    if not token:
        print("ERROR: LOGFIRE_TOKEN is empty or None!")
    logfire.configure(token=token)
    LOGFIRE_STATUS = "Connected & Tracing"
except Exception as e:
    print(f"Logfire Init Error in UI: {e}")
    LOGFIRE_STATUS = f"Standby (Error: {e})"


st.set_page_config(
    page_title="IP-SAKTI Sahayak",
    page_icon="🏛️",
    layout="wide",
)

AI_AVATAR = "🏛️"
USER_AVATAR = "👤"


def format_optional_percent(value):
    if value is None:
        return "Not available"
    try:
        return f"{float(value):.0%}"
    except (TypeError, ValueError):
        return "Not available"


def format_optional_bool(value):
    if value is True:
        return "Yes"
    if value is False:
        return "No"
    return "Needs more information"


if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    logfire.info(f"✨ New User Session Created: {st.session_state.session_id}")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "jurisdiction" not in st.session_state:
    st.session_state.jurisdiction = "INDIA"

if "show_classifier" not in st.session_state:
    st.session_state.show_classifier = False


with st.sidebar:
    st.title("🏛️ IP-SAKTI Sahayak")
    st.caption("Ayurveda IP & Regulatory Guidance")
    st.markdown("---")
    
    st.success(f"Logfire: {LOGFIRE_STATUS}")
    st.info(f"Session: {st.session_state.session_id[:8]}")
    
    st.markdown("### ⚖️ Jurisdiction")
    st.session_state.jurisdiction = st.radio(
        "Select jurisdiction scope:",
        ["INDIA", "INTERNATIONAL", "BOTH"],
        index=0,
        help="Routes queries to the correct legal framework"
    )
    
    st.markdown("---")
    st.markdown("### 🧪 Formulation Classifier")
    if st.button("🔍 Classify a Product", width="stretch"):
        st.session_state.show_classifier = not st.session_state.show_classifier
    
    st.markdown("---")
    st.markdown("### 🌐 Language (Bhashini)")
    language_options = {
        "English": "en", "Hindi": "hi", "Tamil": "ta", "Telugu": "te",
        "Kannada": "kn", "Malayalam": "ml", "Gujarati": "gu", "Marathi": "mr",
        "Bengali": "bn", "Punjabi": "pa", "Odia": "or", "Assamese": "as",
        "Urdu": "ur", "Sanskrit": "sa", "Nepali": "ne", "Konkani": "kok",
        "Maithili": "mai", "Manipuri": "mni", "Sindhi": "sd", "Dogri": "doi",
        "Kashmiri": "ks", "Bodo": "brx"
    }
    selected_language_name = st.selectbox("Interface language:", list(language_options.keys()), index=0)
    language = language_options[selected_language_name]
    st.caption(f"Selected: {selected_language_name} ({language})")
    
    st.markdown("---")
    if st.button("🗑️ Clear History", width="stretch", type="primary"):
        logfire.warn(f"🗑️ Memory Wipe: {st.session_state.session_id}")
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()


st.title("🏛️ IP-SAKTI Sahayak")
st.caption("Ayurveda IP & Regulatory Guidance • Multilingual • Source-Cited")


st.warning(
    "**⚠️ Disclaimer:** This system provides general informational guidance based on "
    "publicly available legal and regulatory sources. It does not constitute legal advice. "
    "For specific cases, consult a qualified IP attorney or regulatory expert.",
    icon="⚠️"
)


if st.session_state.show_classifier:
    with st.expander("🧪 Formulation Classification Wizard", expanded=True):
        st.markdown("Answer a few questions to identify the likely regulatory pathway.")
        
        col1, col2 = st.columns(2)
        with col1:
            desc = st.text_area("Product description:", placeholder="e.g., Herbal tablet for joint pain with 5 classical ingredients...")
            ingredients = st.text_area("Ingredients (optional):", placeholder="Ashwagandha, Guggulu, Shallaki, etc.")
        with col2:
            use = st.selectbox("Intended use:", ["", "Therapeutic (treating disease)", "Health supplement", "Food/Nutraceutical", "Cosmetic/Personal care"])
            classical_ref = st.text_area("Classical text reference (optional):", placeholder="e.g., Charaka Samhita Chikitsasthana 5/24")
        
        if st.button("🎯 Classify", type="primary", width="stretch"):
            if desc:
                with st.status("Classifying...", expanded=True) as status:
                    try:
                        base_url = os.getenv("BACKEND_URL", "http://localhost:8000")
                        response = requests.post(
                            f"{base_url}/classify",
                            json={
                                "description": desc,
                                "ingredients": ingredients,
                                "intended_use": use,
                                "reference_to_classical_text": classical_ref,
                                "language": language
                            },
                            timeout=30
                        )
                        data = response.json()
                        if response.status_code >= 400:
                            detail = data.get("detail") or data.get("error") or response.text
                            raise RuntimeError(f"Backend returned HTTP {response.status_code}: {detail}")
                        status.update(label="✅ Classification Complete", state="complete", expanded=False)
                        
                        pathway = data.get("formulation_type") or "INSUFFICIENT_INFO"
                        st.success(f"**Likely Pathway:** {pathway}")
                        st.info(f"**ABS Required:** {format_optional_bool(data.get('abs_required'))}")
                        st.info(f"**Confidence:** {format_optional_percent(data.get('confidence_score'))}")
                        st.markdown(f"**Explanation:** {data.get('explanation', 'N/A')}")
                        
                    except Exception as e:
                        status.update(label="❌ Error", state="error")
                        st.error(f"Classification failed: {e}")
            else:
                st.warning("Please provide a product description.")


for message in st.session_state.messages:
    avatar = AI_AVATAR if message["role"] == "assistant" else USER_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        
        if message["role"] == "assistant" and "metadata" in message:
            meta = message["metadata"]
            
            with st.expander("🔍 Why Trust This Answer?", expanded=False):
                col1, col2, col3 = st.columns(3)
                intent = meta.get("intent")
                chunks = int(meta.get("chunks_retrieved") or 0)
                confidence = meta.get("confidence_score")
                confidence_applicable = intent not in {"CONVERSATIONAL", "OFF_TOPIC", "BLOCKED"} and confidence is not None

                with col1:
                    st.metric("Jurisdiction", meta.get("jurisdiction") or "Not applicable")
                with col2:
                    st.metric("Chunks Retrieved", chunks)
                with col3:
                    if confidence_applicable:
                        st.metric("Confidence", f"{float(confidence):.0%}")
                    else:
                        st.metric("Confidence", "Not applicable")

                if meta.get("formulation_type"):
                    st.caption(f"Formulation type: {meta['formulation_type']}")
                
                if meta.get("abs_required") is not None:
                    st.info(f"🌿 ABS/Biodiversity compliance: {'Yes' if meta['abs_required'] else 'No'}")

                answer_sets = meta.get("jurisdiction_answer_sets")
                if answer_sets:
                    st.markdown("**Separated Jurisdiction Answer Sets:**")
                    if answer_sets.get("india"):
                        with st.expander("India answer set"):
                            st.markdown(answer_sets["india"])
                    if answer_sets.get("international"):
                        with st.expander("International answer set"):
                            st.markdown(answer_sets["international"])
                    if answer_sets.get("practical_next_steps"):
                        with st.expander("Practical next steps"):
                            st.markdown(answer_sets["practical_next_steps"])

                abs_helper = meta.get("abs_helper")
                if abs_helper:
                    st.markdown("**ABS Helper:**")
                    st.caption(abs_helper.get("status", "not_clearly_indicated"))
                    for step in abs_helper.get("next_steps", []):
                        st.markdown(f"- {step}")

                tkdl_pointer = meta.get("tkdl_prior_art_pointer")
                if tkdl_pointer and tkdl_pointer.get("relevant"):
                    st.markdown("**TKDL / Prior-Art Pointer:**")
                    st.info(tkdl_pointer.get("pointer", "TKDL-aware prior-art review is recommended."))

                escalation = meta.get("escalation")
                if escalation:
                    if escalation.get("recommended"):
                        st.warning(escalation.get("path", "Escalate to a human IP facilitator."))
                    else:
                        st.caption(escalation.get("path", "No immediate escalation trigger detected."))
                
                citations = meta.get("citations", [])
                if citations:
                    st.markdown("**Citations Found:**")
                    for c in citations:
                        source_label = c.get("source", "retrieved context")
                        if c.get("section"):
                            source_label = f"{source_label}, {c['section']}"
                        verified = "verified" if c.get("verified") else "unverified"
                        st.markdown(f"- `{c.get('text', '')}` - *{source_label}* ({verified})")
                elif intent == "CONVERSATIONAL":
                    st.info("Retrieval skipped for this conversational message.")
                elif intent in {"OFF_TOPIC", "BLOCKED"}:
                    st.info("Retrieval skipped because the query was outside the assistant's scope.")
                else:
                    st.info(f"No explicit citations extracted. Based on {chunks} retrieved regulatory chunk(s).")


if prompt := st.chat_input("Ask about Ayurveda IP, regulatory pathways, ABS, GI, or formulation classification..."):
    with logfire.span("💬 User Chat Interaction", user_query=prompt, session_id=st.session_state.session_id):
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar=AI_AVATAR):
            with st.status("🔍 IP-SAKTI is analyzing...", expanded=True) as status:
                try:
                    with logfire.span("📡 Calling RAG Backend"):
                        base_url = os.getenv("BACKEND_URL", "http://localhost:8000")
                        url = f"{base_url}/query"
                        payload = {
                            "q": prompt, 
                            "thread_id": st.session_state.session_id,
                            "language": language,
                            "jurisdiction": st.session_state.jurisdiction,
                        }
                        response = requests.post(url, json=payload, timeout=90)
                        data = response.json()
                        if response.status_code >= 400:
                            detail = data.get("detail") or data.get("error") or response.text
                            raise RuntimeError(f"Backend returned HTTP {response.status_code}: {detail}")
                    
                    steps = data.get("thought_process", [])
                    for step in steps:
                        st.write(f"⚙️ {step}")
                    
                    status.update(label="✅ Answer Synthesized", state="complete", expanded=False)
                    
                    if data.get("sources"):
                        with st.expander("📄 View Retrieved Context (Sources)"):
                            for i, source in enumerate(data["sources"]):
                                preview = source[:120].replace("\n", " ") + "..."
                                with st.expander(f"Source {i+1}: {preview}"):
                                    st.info(source)
                                    
                except Exception as e:
                    logfire.error(f"❌ UI-Backend Connection Failed: {e}")
                    status.update(label="❌ Connection Failed", state="error")
                    st.error("Backend offline. Start the API server.")
                    st.stop()

            answer = data.get("answer") or data.get("error") or "No response."
            
            answer_placeholder = st.empty()
            curr_text = ""
            for char in answer:
                curr_text += char
                answer_placeholder.markdown(curr_text + "▌")
                time.sleep(0.003)
            
            answer_placeholder.markdown(answer)
            
            metadata = {
                "intent": data.get("intent"),
                "jurisdiction": data.get("jurisdiction"),
                "formulation_type": data.get("formulation_type"),
                "citations": data.get("citations", []),
                "confidence_score": data.get("confidence_score"),
                "abs_required": data.get("abs_required"),
                "abs_helper": data.get("abs_helper"),
                "tkdl_prior_art_pointer": data.get("tkdl_prior_art_pointer"),
                "jurisdiction_answer_sets": data.get("jurisdiction_answer_sets"),
                "escalation": data.get("escalation"),
                "chunks_retrieved": data.get("chunks_retrieved", len(data.get("sources", [])))
            }
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer,
                "metadata": metadata
            })
            logfire.info("✅ Chat cycle completed successfully.")
