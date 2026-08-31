from app.observability.logfire_compat import logfire
from app.agents.state import AgentState
from app.gateway import portkey_client, extract_cache_status
from app.config import settings


def generate_node(state: AgentState):
    """
    IP-SAKTI Sahayak Responder:
    Generates source-cited answers for Ayurveda IP and regulatory queries.
    Handles empty context gracefully (degraded mode).
    Always includes standing disclaimer for regulatory responses.
    """
    query = state["current_query"]
    intent = state.get("intent", "")
    jurisdiction = state.get("jurisdiction", "")
    formulation_type = state.get("formulation_type", "")
    documents = state.get("documents", [])

    history_str = ""
    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"

    user_msg = state["messages"][-1]["content"] if state["messages"] else ""

    standing_disclaimer = "\n\n---\n*Disclaimer: This is general information, not legal advice. Consult a qualified IP attorney or regulatory expert for specific cases.*"

    # Check for degraded mode
    degraded = len(documents) == 0 and intent != "CONVERSATIONAL"

    if intent == "CONVERSATIONAL":
        logfire.info("Generating conversational response using memory.")
        prompt = f"""
        You are IP-SAKTI Sahayak, a friendly and helpful AI assistant for Ayurveda IP and regulatory guidance.
        Answer the user's latest message using the CONVERSATION HISTORY below.

        CONVERSATION HISTORY:
        {history_str}

        LATEST MESSAGE:
        "{user_msg}"
        
        Keep responses warm, concise, and helpful. If the user asks about your capabilities, mention:
        - Patent and GI guidance for Ayurveda formulations
        - Regulatory pathway classification (Classical, Proprietary, Phytopharmaceutical, Food, Cosmetic)
        - ABS and biodiversity compliance
        - Traditional knowledge and prior-art awareness
        - Multilingual support for 22 Indic languages
        """
        response_text = _generate_response(prompt, state)
        return _build_response(state, response_text, add_disclaimer=False)

    if intent == "FORMULATION_CLASSIFICATION":
        logfire.info("Generating formulation classification guidance.")
        formulation_guidance = _get_formulation_guidance(formulation_type)
        prompt = f"""
        You are IP-SAKTI Sahayak. The user asked about classifying an Ayurveda product/formulation.
        
        Based on the classification analysis, the likely pathway is: {formulation_type or 'Unknown'}
        ABS/biodiversity compliance may be required: {state.get('abs_required', 'Unknown')}
        
        User query:
        "{user_msg}"
        
        Provide clear, practical guidance:
        1. Explain what this classification means in simple terms
        2. Outline the likely regulatory/IP pathway
        3. Mention any ABS or traditional knowledge considerations
        4. Suggest what additional information would help refine the classification
        
        Do NOT give definitive legal conclusions. Use phrases like "likely pathway" and "typically requires."
        """
        response_text = _generate_response(prompt, state) + standing_disclaimer
        return _build_response(state, response_text)

    # Regulatory/IP query
    logfire.info(f"Generating regulatory/IP RAG response (degraded={degraded}).")
    max_context_chars = 25000
    full_context = ""

    for doc in documents:
        if len(full_context) + len(doc) < max_context_chars:
            full_context += doc + "\n\n"
        else:
            logfire.warning("Context truncated to fit Groq TPM limits.")
            break

    if degraded:
        response_text = _safe_abstention_answer(state) + standing_disclaimer
        return _build_response(state, response_text)

    jurisdiction_note = f"\nJurisdiction scope: {jurisdiction}. " if jurisdiction else ""
    answer_shape = """
    Required structure for BOTH jurisdiction:
    ## India answer
    - Give only Indian-law analysis grounded in Indian sources.
    ## International answer
    - Give only treaty/international analysis grounded in international sources.
    ## Practical next steps
    - Explain what to verify next.
    """ if jurisdiction == "BOTH" else """
    Required structure:
    - Keep the answer within the selected jurisdiction.
    - Do not blend India and international rules unless the selected jurisdiction is BOTH.
    """

    prompt = f"""
    You are IP-SAKTI Sahayak, an AI assistant specializing in Ayurveda intellectual property and regulatory guidance.
    
    INSTRUCTIONS:
    - Answer based ONLY on the provided LEGAL/REGULATORY CONTEXT
    - Cite specific sections, rules, or articles when making claims
    - Use bracketed citations like [Section 3(p), Patents Act] when referencing sources
    - If the answer is not in the context, say so clearly
    - For international queries, distinguish Indian law from other frameworks
    - Do NOT mix domestic Indian and international law inappropriately
    - If context is empty, state that clearly and provide general guidance only
    - Include an ABS/compliance note when biological resources, traditional knowledge, export, foreign access, or commercial utilization are involved
    - Include a TKDL/prior-art pointer when patents, novelty, traditional knowledge, or classical formulations are involved; do not claim to search TKDL
    - Include a human escalation note when facts are incomplete, confidence is low, foreign filing is involved, or a legal decision is needed
    {jurisdiction_note}
    {answer_shape}
    
    LEGAL/REGULATORY CONTEXT:
    {full_context}
    
    CONVERSATION HISTORY:
    {history_str}
    
    USER QUESTION:
    "{user_msg}"
    
    Provide a clear, accurate, and source-cited answer.
    """

    response_text = _ensure_source_summary(_generate_response(prompt, state), documents) + standing_disclaimer
    return _build_response(state, response_text)


def _generate_response(prompt: str, state: AgentState) -> str:
    with logfire.span("✍️ LLM Synthesis"):
        last_error = None
        try:
            response = portkey_client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            last_error = e
            logfire.warning(f"Primary LLM generation failed: {e}")

        fallback_model = settings.GROQ_FALLBACK_MODEL
        if fallback_model and fallback_model != settings.GROQ_MODEL:
            try:
                response = portkey_client.chat.completions.create(
                    model=fallback_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1
                )
                return response.choices[0].message.content
            except Exception as e:
                last_error = e
                logfire.warning(f"Fallback LLM generation failed: {e}")

        logfire.error(f"LLM Generation unavailable; using grounded template fallback: {last_error}")
        return _grounded_template_answer(state)


def _build_response(state: AgentState, content: str, add_disclaimer: bool = True) -> dict:
    answer_sets = _split_jurisdiction_answer_sets(content, state.get("jurisdiction"))
    return {
        "final_answer": content,
        "status": "Response generated." if not state.get("documents") else "Response generated from retrieved context.",
        "plan": state["plan"] + ["Response synthesized"],
        "messages": [{"role": "assistant", "content": content}],
        "abs_helper": _derive_abs_helper(state, content),
        "tkdl_prior_art_pointer": _derive_tkdl_pointer(state, content),
        "jurisdiction_answer_sets": answer_sets,
        "escalation": _derive_escalation(state, content),
    }


def _get_formulation_guidance(formulation_type: str) -> str:
    guidance = {
        "CLASSICAL_AYURVEDIC": "Based on authoritative Ayurvedic texts listed in the Drugs & Cosmetics Act First Schedule. Generally not patentable for the formulation itself, but process patents may be possible.",
        "PROPRIETARY_AYURVEDIC": "A modified or new combination requiring licensing under the proprietary pathway. May be eligible for patent protection if novelty and inventive step can be demonstrated.",
        "PHYTOPHARMACEUTICAL": "Plant-based drug pathway under D&C Rules. Requires standardization, clinical evidence, and specific FDA approval. Distinct from classical Ayurvedic route.",
        "FOOD_AYURVEDA_AAHAR": "FSSAI Ayurveda Aahar regulations apply. Food-grade pathway for nutraceuticals. Lower regulatory burden but no patent protection for the formulation.",
        "COSMETIC": "Cosmetic Rules apply. No therapeutic claims allowed. Trademark and design protection may be relevant.",
        "INSUFFICIENT_INFO": "Need more details to classify. Key factors: ingredients, intended use, manufacturing process, reference to classical texts."
    }
    return guidance.get(formulation_type, guidance["INSUFFICIENT_INFO"])


def _grounded_template_answer(state: AgentState) -> str:
    query = (state.get("current_query") or "").lower()
    documents = state.get("documents", []) or []
    intent = state.get("intent", "")

    if intent == "CONVERSATIONAL":
        return (
            "Hello, I am IP-SAKTI Sahayak. I can help with Ayurveda patentability, "
            "GI protection, ABS and biodiversity compliance, FSSAI or AYUSH pathways, "
            "and source-cited regulatory guidance."
        )

    if intent == "FORMULATION_CLASSIFICATION":
        formulation_type = state.get("formulation_type") or "INSUFFICIENT_INFO"
        guidance = _get_formulation_guidance(formulation_type)
        abs_text = state.get("abs_required")
        return (
            f"Likely pathway: {formulation_type}. {guidance}\n\n"
            f"ABS/biodiversity compliance indicator: {abs_text if abs_text is not None else 'needs more information'}.\n\n"
            "This is a preliminary classification. Ingredients, intended use, claims, manufacturing process, "
            "and any classical-text reference would help refine it."
        )

    if not documents:
        return (
            "I could not retrieve authorized source context for this query right now. "
            "For Ayurveda IP or regulatory questions, please ask with the relevant statute, product type, "
            "ingredient, or jurisdiction so I can ground the answer in the approved corpus."
        )

    source_lines = _source_lines(documents)
    if "ipc25" in query or "ipc 25" in query or "section 25" in query:
        return (
            "If by “IPC25” you mean Section 25 in the Indian patents context, it is relevant to patent opposition. "
            "For Ayurveda or biological-resource inventions, the bundled source notes that failure to disclose "
            "biological material or associated traditional knowledge can become a ground for opposition under "
            "Section 25 and revocation under Section 64.\n\n"
            "In practical terms, an Ayurveda patent application should disclose the geographical origin of biological "
            "material, associated traditional knowledge if any, and ABS or prior-informed-consent arrangements where applicable.\n\n"
            f"Sources used:\n{source_lines}"
        )

    if "ipc" in query:
        return (
            "In this Ayurveda IP assistant, “IPC” may refer either to patent classification language or to a shorthand "
            "for an Indian patent-law provision. The retrieved authorized context supports guidance on patentability, "
            "traditional knowledge, and biological-source disclosure, but it does not define a standalone term called “IPC” clearly.\n\n"
            f"Sources used:\n{source_lines}"
        )

    return (
        "Based on the authorized retrieved context, Ayurveda formulations documented as traditional knowledge can face "
        "a patentability bar, while novel processes or genuinely new proprietary formulations may require separate novelty, "
        "inventive-step, licensing, and ABS checks. For biological resources or traditional knowledge, disclose source/origin "
        "and compliance status in the patent workflow.\n\n"
        f"Sources used:\n{source_lines}"
    )


def _source_lines(documents: list[str]) -> str:
    lines = []
    for doc in documents[:5]:
        source = "authorized retrieved context"
        section = None
        for line in doc.splitlines()[:8]:
            if line.startswith("SOURCE:"):
                source = line.replace("SOURCE:", "", 1).strip()
            elif line.startswith("SECTION:"):
                section = line.replace("SECTION:", "", 1).strip()
        label = f"- {source}"
        if section:
            label += f", {section}"
        lines.append(label)
    return "\n".join(lines) if lines else "- authorized retrieved context"


def _ensure_source_summary(answer: str, documents: list[str]) -> str:
    if not documents:
        return answer
    if "sources used:" in answer.lower():
        return answer
    return f"{answer}\n\nSources used:\n{_source_lines(documents)}"


def _safe_abstention_answer(state: AgentState) -> str:
    return (
        "I could not retrieve authorized source context for this question, so I should not give a legal or regulatory answer as if it were grounded.\n\n"
        "Please retry after corpus ingestion is available, or narrow the query by adding the statute, rule, product type, ingredient, registry record, case, or jurisdiction you want checked.\n\n"
        "Escalation: take this to a human IP facilitator or qualified IP/regulatory professional before making a filing, commercialization, export, or ABS decision."
    )


def _split_jurisdiction_answer_sets(content: str, jurisdiction: str | None) -> dict | None:
    if jurisdiction != "BOTH":
        return None
    lower = content.lower()
    india_marker = "## india answer"
    intl_marker = "## international answer"
    if india_marker not in lower or intl_marker not in lower:
        return {
            "india": content,
            "international": "International analysis was not separately generated; rerun with an explicit international query.",
        }
    india_start = lower.find(india_marker)
    intl_start = lower.find(intl_marker)
    next_steps_start = lower.find("## practical next steps")
    india_text = content[india_start:intl_start].strip()
    intl_end = next_steps_start if next_steps_start != -1 else len(content)
    international_text = content[intl_start:intl_end].strip()
    practical = content[next_steps_start:].strip() if next_steps_start != -1 else ""
    return {"india": india_text, "international": international_text, "practical_next_steps": practical}


def _derive_abs_helper(state: AgentState, content: str) -> dict:
    text = f"{state.get('current_query', '')}\n{content}".lower()
    triggers = [
        "biological resource", "biodiversity", "abs", "benefit sharing", "prior informed consent",
        "foreign", "export", "traditional knowledge", "plant", "herb", "ashwagandha", "neem", "turmeric",
    ]
    indicated = bool(state.get("abs_required")) or any(term in text for term in triggers)
    return {
        "required_or_possible": indicated,
        "status": "possible_abs_review_needed" if indicated else "not_clearly_indicated",
        "next_steps": [
            "Identify biological resource, source location, supplier, and community/TK links.",
            "For India, check whether SBB prior intimation or NBA approval is triggered.",
            "Document benefit-sharing, consent, and source/origin evidence before filing or commercialization.",
        ] if indicated else [
            "No clear ABS trigger was detected from the current facts.",
            "Reassess if biological resources, TK, export, foreign access, or commercial utilization are added.",
        ],
    }


def _derive_tkdl_pointer(state: AgentState, content: str) -> dict:
    text = f"{state.get('current_query', '')}\n{content}".lower()
    relevant = any(term in text for term in ("tkdl", "traditional knowledge", "classical", "prior art", "patent", "novelty"))
    return {
        "relevant": relevant,
        "pointer": (
            "TKDL-aware prior-art screening is recommended. This prototype can point to public TKDL/prior-art guidance "
            "from authorized sources, but it must not claim to search the restricted TKDL database directly."
        ) if relevant else "No TKDL/prior-art trigger was detected from the current facts.",
    }


def _derive_escalation(state: AgentState, content: str) -> dict:
    confidence = state.get("confidence_score")
    documents = state.get("documents", []) or []
    needs_human = (
        not documents
        or confidence is None
        or confidence < 0.65
        or state.get("jurisdiction") in {"BOTH", "INTERNATIONAL"}
        or any(term in content.lower() for term in ("not enough", "uncertain", "consult", "qualified"))
    )
    return {
        "recommended": needs_human,
        "path": "Escalate to a human IP facilitator or qualified IP/regulatory professional before filing, commercialization, foreign transfer, or legal correspondence.",
        "reasons": [
            "Source coverage or confidence is limited.",
            "The query may involve jurisdiction-specific legal judgment.",
            "The assistant provides information only, not legal advice.",
        ] if needs_human else ["No immediate escalation trigger detected, but professional review is still recommended for action decisions."],
    }
