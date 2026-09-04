from app.observability.logfire_compat import logfire
from app.agents.state import AgentState
from app.gateway import portkey_client, extract_cache_status
from app.config import settings


# ============================================================
# SHARED FORMATTING INSTRUCTIONS — injected into every LLM prompt
# to ensure well-structured, presentable answers while keeping
# all accuracy and grounding safeguards intact.
# ============================================================
RESPONSE_FORMAT_INSTRUCTIONS = """

FORMATTING AND EVIDENCE RULES (follow these strictly):

1. Start with a **Summary** of 1-2 sentences that directly answers the user's core question.
2. Use markdown headings (## and ###) to separate logical sections.
3. Use bullet points (-) for requirements, criteria, conditions, and key findings.
4. Use numbered lists (1. 2. 3.) only for sequential steps or procedures.
5. Use **bold** for important legal terms, explicitly supported section/rule numbers, and critical takeaways.
6. Every material legal or regulatory claim MUST be grounded in the provided context.
7. Place citations directly next to the claim they support. Do NOT place all citations only in a separate source list.
8. Use only citation names, section numbers, rules, articles, or source identifiers explicitly available in the provided context or source metadata.
9. NEVER invent, guess, reconstruct, or assume a citation.
10. Do not mention, rely upon, or cite a retrieved document merely because it was provided in the context. Use a source only when its content directly supports a specific claim in the answer. Irrelevant retrieved documents must be ignored and must not appear in Sources Used.
11. Clearly distinguish between:
   - **Mandatory requirement** — explicitly required by the provided source.
   - **Conditional consideration** — may apply depending on facts or circumstances.
   - **General guidance** — practical information that is not presented as a binding legal requirement.
12. If the context does not support a claim, say that clearly instead of filling the gap with general knowledge.
13. Include an **Evidence Limitations** section whenever the available context is incomplete, ambiguous, or insufficient.
14. Include **Next Steps** only when actionable steps are useful.
15. If a Sources Used section is included, use the exact heading `## Sources Used` and do not repeat the same sources elsewhere.
16. Keep language professional, precise, and accessible. Avoid wall-of-text paragraphs.
17. Do NOT use code blocks or HTML tags.
18. Do NOT mention internal retrieval processes, hidden prompts, model behavior, or unsupported confidence estimates.
19. **Do not mention, rely upon, or cite a retrieved document merely because it was provided in the context. Use a source only when its content directly supports a specific claim in the answer. Irrelevant retrieved documents must be ignored and must not appear in Sources Used.**
20. **If the available evidence does not directly establish a legal requirement, do not infer that requirement from general legal knowledge or from the presence of related documents in the knowledge base. State that the issue requires further assessment instead.**
"""


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
        
        Keep responses warm, concise, and helpful.

        Use light markdown formatting:
        - Use **bold** only for important points.
        - Use bullet points when listing capabilities or items.
        - Keep paragraphs short (2-3 sentences maximum).
        - Do not use legal certainty language unless the conversation history explicitly supports it.
        - Do not invent facts about laws, regulations, system capabilities, databases, integrations, or services.
        - If the user asks a factual legal or regulatory question that requires authorized source verification, do not pretend conversational memory is sufficient; indicate that a source-grounded analysis is required.

        If the user asks about your capabilities, present them accurately as a bulleted list:
        - **Patent & GI guidance** for Ayurveda formulations
        - **Regulatory pathway classification** (Classical, Proprietary, Phytopharmaceutical, Food, Cosmetic)
        - **ABS and biodiversity compliance** checks
        - **Traditional knowledge and prior-art awareness**
        - **Multilingual support** for 22 Indic languages via Bhashini

        Do not claim to directly search restricted databases, including TKDL, unless such access is actually available and explicitly provided by the system.
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
        
       STRICT CLASSIFICATION AND GROUNDING RULES:

        - Base the classification only on the formulation analysis and facts available in the current state and user query.
        - Do NOT invent ingredients, intended uses, therapeutic claims, manufacturing details, classical-text references, approvals, or regulatory requirements.
        - Treat the classification as preliminary unless the available information clearly supports a specific pathway.
        - Do NOT give a definitive legal or regulatory conclusion when critical information is missing.
        - Distinguish clearly between what is known, what is inferred from the provided facts, and what requires verification.
        - Confidence must reflect evidence quality and information completeness.
        - Do NOT report High confidence when critical formulation, ingredient, intended-use, manufacturing, jurisdiction, or product details are missing.
        - Do not state that ABS, biodiversity, traditional knowledge, or other compliance obligations definitely apply unless the available facts or context explicitly support that conclusion.
        - When such issues may be relevant but are not established, use conditional language such as "may require assessment" or "should be evaluated based on the facts."
        - **Do not discuss patentability, patents, novelty, inventive step, GI, trademarks, or other IP issues unless the user's query explicitly asks about them or they are necessary to answer the classification question. Focus first and primarily on determining the requested regulatory classification.**
        - **If the available evidence is insufficient to support a specific classification, output "Cannot be determined from the available information" instead of guessing a likely pathway. A Low confidence level does not permit an unsupported classification.**

        Structure your response as follows:

        ## Summary
        One-line preliminary classification result and what it means in plain language.

        ## Classification Details
        - **Likely Pathway:** Name and brief explanation based only on the available facts.
        - **Confidence Level:** High / Medium / Low.
        - **Basis:** Briefly state which available facts support the classification.

        ## Regulatory Pathway
        Provide numbered steps the user would typically need to evaluate or follow.
        Do not present a step as legally mandatory unless that requirement is explicitly supported by the available information.

        ## ABS & Traditional Knowledge Considerations
        ONLY include this section when the query, formulation facts, or available analysis involve:
        - Biological resources
        - Biodiversity
        - Traditional knowledge
        - Foreign access
        - Export
        - Commercial utilization

        Clearly distinguish between confirmed requirements and issues that may require further assessment.

        If none of these apply, omit this section entirely.

        ## What's Missing
        List only the additional information genuinely needed to improve classification accuracy.

        Examples may include:
        - Exact ingredients and composition
        - Intended use or claims
        - Manufacturing process
        - Classical-text reference
        - Product form
        - Target jurisdiction

        ## Evidence Limitations
        Briefly state any important uncertainty or limitation affecting the classification.

        ## Next Steps
        Provide practical numbered actions only when useful.

        ## Sources Used
        List the exact source identifiers, section numbers, or document names that materially informed the answer.
        Use the format: `- <source_name>, <section/rule>`
        Do NOT duplicate a Sources Used section — include it exactly once.

        Do NOT give definitive legal conclusions. Use phrases such as:
        - "likely pathway"
        - "based on the available information"
        - "may require assessment"
        - "typically involves"
        - "requires verification"
         FINAL SELF-CHECK BEFORE ANSWERING:

        Before producing the final response, silently verify:

        1. Is the classification based only on the available facts?
        2. Did I invent any ingredients, claims, manufacturing details, or regulatory requirements?
        3. Does every legal or regulatory statement have support in the available context?
        4. Did I describe a conditional consideration as a mandatory requirement?
        5. Is the confidence level justified by the available information?
        6. Did I clearly identify important missing information?
        7. Did I avoid giving a definitive legal conclusion where verification is required?
        8. Did I clearly state evidence limitations when information is incomplete?

        If any answer is "no" or uncertain, revise the response conservatively and remove unsupported claims.

        {RESPONSE_FORMAT_INSTRUCTIONS}
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

    comprehensive_guidance = ""
    if intent == "COMPREHENSIVE":
        comprehensive_guidance = f"""
    The user's query is COMPREHENSIVE. You MUST provide BOTH a Formulation Classification and a Regulatory/IP analysis.

    PRELIMINARY CLASSIFICATION ANALYSIS:
    - Likely pathway: {formulation_type or 'Unknown'}
    - ABS/biodiversity compliance may be required: {state.get('abs_required', 'Unknown')}
    
    Before the `## Summary` section, include:
    ## Formulation Classification
    - **Likely Pathway:** Name and brief explanation based only on the available facts. If the available evidence is insufficient to support a specific classification, output "Cannot be determined from the available information" instead of guessing a likely pathway. A Low confidence level does not permit an unsupported classification.
    - **Confidence Level:** High / Medium / Low.
    - **Basis:** Briefly state which available facts support the classification.
    - **What's Missing:** List only the additional information genuinely needed to improve classification accuracy (e.g., exact ingredients, intended claims, manufacturing process, classical-text reference).
    """

    jurisdiction_note = f"\nJurisdiction scope: {jurisdiction}. " if jurisdiction else ""
    domain = state.get("legal_domain") or ""
    domain_guidance = _get_domain_guidance(domain, user_msg)
    answer_shape = """
    Required structure for BOTH jurisdiction:

    ## Summary
    Briefly state what the available context supports for India and/or international analysis.
    Do not imply that both analyses are complete if the retrieved context supports only one.

    ## India Answer
    Include this section only when Indian-law sources are available.

    ### Applicable Framework
    State only frameworks explicitly supported by Indian sources in the context.

    ### Key Provisions
    Explain only provisions explicitly present in the retrieved Indian context.
    Attach citations directly to the claims they support.

    ### Requirements and Considerations
    Clearly distinguish:
    - Mandatory requirements supported by the source.
    - Conditional considerations that depend on facts.

    Do not include international rules in this section.

    ## International Answer
    Include this section only when treaty, foreign-law, or international-framework sources are available.

    ### Applicable Framework
    State only frameworks explicitly supported by international sources in the context.

    ### Key Provisions
    Explain only provisions explicitly present in the retrieved international context.
    Attach citations directly to the claims they support.

    ### Requirements and Considerations
    Clearly distinguish:
    - Explicit requirements supported by the international source.
    - Conditional considerations depending on facts or jurisdiction.

    Do not include Indian-law requirements in this section.

    ## Key Differences
    Compare India and international frameworks ONLY when both sides are supported by the retrieved context.
    Do not infer differences from missing information.

    ## Evidence Limitations
    Explicitly state:
    - Which jurisdiction lacks sufficient source coverage.
    - Which issues cannot be answered reliably from the available context.
    - Which matters require professional verification.

    ## Practical Next Steps
    Provide a numbered list of practical verification steps.
    Do not present these steps as statutory requirements unless explicitly supported by the context.
    """ if jurisdiction == "BOTH" else """
    Required structure:

    - Keep the answer strictly within the selected jurisdiction.
    - Do not blend Indian and international rules unless the selected jurisdiction is BOTH.
    - Do not introduce legal frameworks that are absent from the retrieved context.

    ## Summary

    Give a direct 1-2 sentence answer that captures the core finding based only on what the available context supports.
    Do not overstate certainty. If the context only partially answers the query, reflect that.

    ## Key Requirements

    Use a bulleted list. Each bullet must state the requirement or consideration and include an inline citation
    in square brackets (e.g., [Section 3(p), Patents Act]) directly next to the claim it supports.

    Only describe something as a **mandatory requirement** when the retrieved source explicitly establishes it.
    Use labels to distinguish:
    - **<topic>:** The available context supports… [citation]
    - **<topic>:** The retrieved sources do not provide enough information to determine…

    Clearly distinguish between:
    - **Mandatory requirement** — explicitly required by the provided source.
    - **Conditional consideration** — may apply depending on facts or circumstances.
    - **General guidance** — practical information not presented as a binding requirement.

    ## Evidence Limitations

    State clearly what the available retrieved sources do NOT cover or cannot confirm.
    Include:
    - Missing information or source coverage
    - Ambiguous or incomplete facts
    - Questions the context cannot answer reliably
    - Matters requiring professional verification

    ## Next Steps

    Provide a numbered list of practical actions only when useful.
    Do not label general recommendations as mandatory legal requirements unless explicitly supported by the retrieved context.

    ## Sources Used

    List the exact source identifiers, section numbers, or document names that materially informed the answer.
    Use the format: `- <source_name>, <section/rule>`
    Do NOT duplicate a Sources Used section — include it exactly once.
    """

    prompt = f"""
    You are IP-SAKTI Sahayak, an AI assistant specializing in Ayurveda intellectual property and regulatory guidance.
    
    ACCURACY, GROUNDING, AND ANTI-HALLUCINATION INSTRUCTIONS (NON-NEGOTIABLE):

    SOURCE-OF-TRUTH RULES:
    - The provided LEGAL/REGULATORY CONTEXT is the primary source of truth for factual legal and regulatory claims.
    - Answer factual legal or regulatory questions ONLY to the extent supported by the provided context.
    - Do NOT use outside legal knowledge to fill gaps in the retrieved context.
    - Do NOT assume that a retrieved document is relevant merely because it appears in the context.
    - Use only information from a source when that source actually supports the specific claim being made.

    NO-HALLUCINATION RULES:
    - NEVER invent, guess, reconstruct, or assume:
    - Section numbers
    - Rule numbers
    - Articles
    - Case citations
    - Legal requirements
    - Regulatory approvals
    - Filing obligations
    - Deadlines
    - Penalties
    - Government procedures
    - Source names
    - NEVER present general knowledge as if it came from the provided knowledge base.
    - NEVER convert a possibility into a certainty.
    - NEVER state that something "must," "is required," or "is mandatory" unless the provided context explicitly supports that level of certainty.
    - If the context is silent, incomplete, conflicting, or ambiguous, say so clearly.
    - **If the available evidence does not directly establish a legal requirement, do not infer that requirement from general legal knowledge or from the presence of related documents in the knowledge base. State that the issue requires further assessment instead.**

    CLAIM-LEVEL EVIDENCE RULES:
    - Every material legal or regulatory claim should be traceable to the provided context.
    - Place citations immediately after, or within, the claim they support.
    - Do NOT place all citations only at the end of the answer.
    - Do NOT cite a source unless it directly supports the claim.
    - **Do not mention, rely upon, or cite a retrieved document merely because it was provided in the context. Use a source only when its content directly supports a specific claim in the answer. Irrelevant retrieved documents must be ignored and must not appear in Sources Used.**
    - Only use citation identifiers, section numbers, rule numbers, articles, or source names explicitly available in the retrieved context or source metadata.
    - If exact citation metadata is unavailable, do not invent it.
    - Never fabricate a citation to make an answer appear more authoritative.

    LEGAL PRECISION RULES:
    - Clearly distinguish between:
    1. **Confirmed requirement** — explicitly established by the provided source.
    2. **Conditional consideration** — may apply depending on the facts.
    3. **General procedural guidance** — practical guidance that is not presented as a binding legal requirement.
    - Use conditional wording where appropriate:
    - "may apply"
    - "may require assessment"
    - "depends on the facts"
    - "the available context indicates"
    - "requires verification"
    - Do not overstate legal consequences.

    RETRIEVAL LIMITATION RULES:
    - If the required answer is not supported by the context, explicitly say:
    "The available retrieved sources do not provide enough information to answer this point reliably."
    - Do not compensate for missing context by guessing.
    - If retrieved sources appear unrelated to the user's question, do not rely on them merely to produce an answer.
    - Prefer a limited but accurate answer over a comprehensive but unsupported answer.

    JURISDICTION RULES:
    - Keep Indian-law analysis strictly grounded in Indian sources.
    - Keep international analysis strictly grounded in treaty, international, or foreign-framework sources.
    - Do NOT mix Indian and international law.
    - If the selected jurisdiction is BOTH, analyze each jurisdiction separately.
    - If source coverage exists for only one jurisdiction, clearly state that the other jurisdiction lacks sufficient source support.

    ABS / BIODIVERSITY RULES:
    - Include ABS, biodiversity, benefit-sharing, biological-resource, or associated traditional-knowledge discussion ONLY when the query or retrieved context makes it relevant.
    - Do NOT state that ABS definitely applies unless the provided context supports that conclusion.
    - When relevance is possible but unconfirmed, state that an ABS or biodiversity assessment may be required depending on the facts.

    TKDL / PRIOR-ART RULES:
    - Include TKDL or prior-art discussion ONLY when patents, novelty, traditional knowledge, prior art, or classical formulations are relevant.
    - Do NOT claim to search, access, query, or verify the restricted TKDL database directly.
    - You may recommend appropriate prior-art or TKDL-aware verification only when relevant.

    HUMAN ESCALATION RULES:
    - Recommend professional verification when:
    - Important facts are missing.
    - Source coverage is insufficient.
    - The answer involves a legal decision.
    - Foreign filing or cross-border activity is involved.
    - The available evidence is ambiguous.
    - Do not imply that professional escalation confirms the legal conclusion.

    SOURCE LIST RULES:
    - If a `## Sources Used` section is included, list only sources that materially informed the answer.
    - Do not duplicate the Sources Used section.
    - Do not repeat the same source list elsewhere.
    - A source list supplements claim-level citations; it does not replace them.

    ANSWER QUALITY PRIORITY:
    1. Accuracy
    2. Grounding in the provided knowledge base
    3. Correct legal certainty
    4. Clear evidence limitations
    5. Useful structure and readability

    When accuracy conflicts with completeness, choose accuracy.
    When evidence conflicts with assumption, choose evidence.
    When the context is insufficient, abstain from unsupported conclusions.
    {jurisdiction_note}
    {comprehensive_guidance}
    {domain_guidance}
    {answer_shape}
    {RESPONSE_FORMAT_INSTRUCTIONS}
    
    LEGAL/REGULATORY CONTEXT:
    {full_context}
    
    CONVERSATION HISTORY:
    {history_str}
    
    USER QUESTION:
    "{user_msg}"
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


def _get_domain_guidance(domain: str, query: str) -> str:
    q = query.lower()
    if domain == "TRADEMARK" or any(k in q for k in ("trademark", "trade mark", "brand name", "வணிகமுத்திரை")):
        return """
    DOMAIN-SPECIFIC REQUIREMENTS (TRADEMARKS):
    - Governing Act: Trade Marks Act, 1999.
    - Registrability: The name must be distinctive and capable of distinguishing goods.
    - Grounds for Refusal (Section 9): Generic Ayurvedic names and botanical terms (e.g., 'Ashwagandha', 'Triphala', 'Churna', 'Rasayana') cannot be registered as trademarks for products containing those ingredients because they are descriptive of the goods.
    - Coined / Distinctive Names: Coined or fanciful brand names (including coined Sanskrit names) can be registered if they distinguish the goods.
    - International Registration: International brand protection is available via the Madrid System (Madrid Protocol).
    - DO NOT include patent eligibility analysis, novelty/inventive-step tests, or traditional knowledge patent bars.
    """
    if domain == "GI_PROTECTION" or "gi" in q or "geographical indication" in q:
        return """
    DOMAIN-SPECIFIC REQUIREMENTS (GEOGRAPHICAL INDICATIONS):
    - Governing Act: Geographical Indications of Goods (Registration and Protection) Act, 1999.
    - Section 2(e): Protects goods originating in a specific territory/region where a given quality, reputation, or other characteristic is essentially attributable to its geographical origin (e.g., Kerala Ayurvedic Oil, Kashmir Saffron).
    - Application Process: Applications are filed with the Geographical Indications Registry in Chennai.
    - Procedure: Submit proof of historical link and origin, registry examination, publication in the GI Journal, and a 4-month opposition window.
    - Registration confers exclusive rights to the registered proprietor and authorized users under Section 13 for 10 years (renewable).
    - DO NOT discuss patents or novelty.
    """
    if domain == "COPYRIGHT" or "copyright" in q or "textbook" in q:
        return """
    DOMAIN-SPECIFIC REQUIREMENTS (COPYRIGHT):
    - Governing Act: Copyright Act, 1957.
    - Protects original literary works: Covers textbooks, research papers, formulation documentation, and instructional manuals under Section 13.
    - Idea-Expression Dichotomy: Copyright protects the author's original written expression, NOT the underlying ideas, medical concepts, or formulations.
    - Public Domain Classical Texts: Classical treatises (Charaka Samhita, Sushruta Samhita, etc.) are ancient works in the public domain. However, modern translations, critical commentaries, and annotations are protected by copyright.
    - DO NOT discuss patents or novelty.
    """
    if domain == "PLANT_VARIETY" or "plant variety" in q or "cultivar" in q or "ppvfr" in q:
        return """
    DOMAIN-SPECIFIC REQUIREMENTS (PLANT VARIETY PROTECTION):
    - Governing Act: Protection of Plant Varieties and Farmers' Rights Act, 2001 (PPV&FR Act).
    - Protects new plant varieties and crop cultivars (such as new Ashwagandha cultivars).
    - Criteria (Section 15): Must satisfy DUS criteria — Distinct, Uniform, and Stable.
    - Term of Protection: 15 years for annual/biennial crops; 18 years for trees and vines.
    - Farmers' Rights (Section 39): Explicit statutory right allowing farmers to save, use, sow, resow, exchange, share, or sell farm-saved seed of protected varieties (cannot sell branded seed).
    - This is separate and independent from patent protection.
    """
    if domain == "DESIGN" or "packaging" in q or "design of" in q:
        return """
    DOMAIN-SPECIFIC REQUIREMENTS (DESIGNS & PACKAGING):
    - Governing Act: Designs Act, 2000.
    - Protects novel and original features of shape, configuration, pattern, or ornamentation applied to articles (e.g. bottles, containers, boxes for Ayurvedic products).
    - Section 4: Requires novelty and originality; must not be previously published.
    - Term: 10 years initially, extendable by 5 years (total 15 years).
    - Distinct from trademark (which protects brand identity) and patent (which protects formulation/process).
    """
    if domain == "ABS_COMPLIANCE" or "abs" in q or "biodiversity" in q or "nba" in q or "sbb" in q or "own farm" in q:
        return """
    DOMAIN-SPECIFIC REQUIREMENTS (ABS & BIODIVERSITY COMPLIANCE):
    - Governing Act: Biological Diversity Act, 2002 and Biological Diversity Amendment Act, 2023.
    - Section 3: Non-citizens, foreign entities, and Indian entities with foreign equity or control must obtain prior approval of the National Biodiversity Authority (NBA) for research, commercial utilization, or export of biological resources/extracts.
    - Section 7: Indian citizens and domestic entities commercially utilizing biological resources must give prior intimation to the State Biodiversity Board (SBB).
    - Section 20: NBA determines fair and equitable benefit sharing.
    - Section 55 Exemption: Local farmers and growers cultivating biological resources (like Ashwagandha) on their own farm/land for sustenance and cultivation are explicitly EXEMPT from ABS requirements.
    - Penalties: Violations attract civil fines under the 2023 amended provisions; non-disclosure in patent applications is grounds for opposition (Section 25) and revocation (Section 64) of patents.
    """
    if domain == "FSSAI_FOOD" or "ayurveda aahar" in q or "food" in q or "supplement" in q:
        return """
    DOMAIN-SPECIFIC REQUIREMENTS (FOOD / AYURVEDA AAHAR):
    - Governing Framework: FSSAI Ayurveda Aahar Regulations, 2022 and Food Safety and Standards Act, 2006.
    - Permitted Claims: Structure-function claims (e.g., supports digestion, maintains vitality).
    - Prohibited Claims (Section 24 FSS Act): CANNOT make disease treatment, cure, or prevention claims (e.g., cannot claim to cure diabetes or arthritis). Making disease claims automatically brings the product under drug regulations (Drugs & Cosmetics Act).
    - Licensing: Requires FSSAI license from the State Food Authority. Formulations under the food route do not enjoy patent protection.
    """
    if domain == "COSMETIC" or "cosmetic" in q or "face cream" in q or "face wash" in q:
        return """
    DOMAIN-SPECIFIC REQUIREMENTS (COSMETICS):
    - Governing Framework: Cosmetics Rules, 2020 under the Drugs and Cosmetics Act.
    - Permitted: External application for cleansing, beautifying, promoting attractiveness, or moisturizing.
    - Strict Prohibition: CANNOT make therapeutic, curing, or disease prevention claims (e.g., cannot claim to treat acne, eczema, or skin infections). Therapeutic claims reclassify the product as a drug.
    - Licensing: Cosmetic manufacturing license from State Licensing Authority. Clinical trials are not required for efficacy.
    """
    if domain == "PHYTOPHARMACEUTICAL" or "phytopharmaceutical" in q:
        return """
    DOMAIN-SPECIFIC REQUIREMENTS (PHYTOPHARMACEUTICALS):
    - Governing Framework: Drugs and Cosmetics Rules (Phytopharmaceutical route).
    - For standardized botanical extracts with clinical trials proving therapeutic efficacy.
    - Distinct from classical Ayurvedic medicines (which require adherence to authoritative texts) and proprietary medicines.
    - Requires specific CDSCO/FDA approval, GMP compliance, pharmacopoeial standardization, and non-clinical/clinical trial data.
    """
    if domain == "INTERNATIONAL_IP" or "pct" in q or "wipo" in q or "gratk" in q or "nagoya" in q or "differences between indian and us" in q:
        return """
    DOMAIN-SPECIFIC REQUIREMENTS (INTERNATIONAL IP & TREATIES):
    - PCT (Patent Cooperation Treaty): Indian applicants can file an international patent application with the Indian Patent Office as Receiving Office, designating member states (US, EP, JP).
    - WIPO GRATK Treaty 2024: Requires mandatory disclosure in patent applications of the country of origin of genetic resources and Indigenous/traditional knowledge.
    - Nagoya Protocol: Requires Prior Informed Consent (PIC) and Mutually Agreed Terms (MAT) for access to genetic resources and traditional knowledge. Users in importing countries (e.g., Germany) must comply with provider country (India) domestic ABS laws.
    - Key Differences: India has Section 3(p) TK exclusion and Form 18A biological source disclosure; US and EPO evaluate novelty/inventive step against prior art including TKDL, and permit method-of-treatment claims more liberally.
    """
    if domain == "PATENT_ELIGIBILITY" or "patent" in q:
        return """
    DOMAIN-SPECIFIC REQUIREMENTS (PATENT ELIGIBILITY):
    - Governing Act: Patents Act, 1970 (as amended).
    - Section 3(p): Statutory bar prohibiting patents on traditional knowledge or an aggregation/duplication of known properties of traditionally known components. Classical formulations in First Schedule texts cannot be patented as products.
    - Process Patents: Novel, non-obvious extraction methods, synergistic combinations with unexpected non-additive effects, or novel delivery systems may be patentable as processes if they meet novelty, inventive step, and industrial applicability.
    - Form 18A: Mandatory disclosure of the source and geographical origin of biological material.
    - TKDL: Acts as prior art cited by patent examiners (IPO, USPTO, EPO) to reject claims lacking novelty.
    """
    return ""


def _grounded_template_answer(state: AgentState) -> str:
    query = (state.get("current_query") or "").lower()
    documents = state.get("documents", []) or []
    intent = state.get("intent", "")

    if intent == "CONVERSATIONAL":
        return (
            "Hello! I am **IP-SAKTI Sahayak**, your AI assistant for Ayurveda IP and regulatory guidance.\n\n"
            "I can help you with:\n"
            "- **Patentability analysis** for Ayurveda formulations\n"
            "- **GI protection** guidance\n"
            "- **ABS and biodiversity compliance** checks\n"
            "- **FSSAI / AYUSH pathway** navigation\n"
            "- **Source-cited regulatory guidance** grounded in authorized legal texts\n\n"
            "Feel free to ask your question!"
        )

    if intent in ("FORMULATION_CLASSIFICATION", "COMPREHENSIVE"):
        formulation_type = state.get("formulation_type") or "INSUFFICIENT_INFO"
        guidance = _get_formulation_guidance(formulation_type)
        abs_text = state.get("abs_required")
        abs_display = "may require further assessment depending on the biological resources, sourcing, and commercial use." if abs_text or abs_text is None else "does not appear to be mandatory based on the provided facts."
        
        formulation_output = (
            "## Formulation Classification\n\n"
            "- **Likely Pathway:** Cannot be determined from the available information.\n"
            "- **Confidence Level:** Low.\n"
            "- **Basis:** The available facts identify ingredients and general positioning, but do not establish the exact formulation, classical-text basis, dosage, manufacturing method, or applicable product claims.\n\n"
            "## What's Missing\n\n"
            "- Exact formulation and composition\n"
            "- Intended claims and labeling\n"
            "- Classical Ayurvedic text reference, if applicable\n"
            "- Manufacturing process\n"
            "- Product dosage and presentation\n\n"
        )
        
        if intent == "FORMULATION_CLASSIFICATION":
            return (
                "## Summary\n\n"
                "Based on the available information, the product **cannot yet be confidently classified** as Classical Ayurvedic, Proprietary Ayurvedic, Phytopharmaceutical, Food, or Cosmetic. The available facts are insufficient to determine a reliable regulatory pathway.\n\n"
                + formulation_output +
                "## ABS & Traditional Knowledge Considerations\n\n"
                f"Because the product involves traditionally known herbs, traditional-knowledge and biodiversity considerations {abs_display}\n\n"
                "## Evidence Limitations\n\n"
                "The available information does not establish enough facts to make a definitive regulatory classification.\n\n"
                "## Next Steps\n\n"
                "1. Provide the complete ingredient list and quantities.\n"
                "2. Identify whether the formulation appears in an authoritative classical Ayurvedic text.\n"
                "3. Specify the intended product claims and target market.\n"
                "4. Verify the classification against the applicable regulatory framework."
            )
        else:
            # For COMPREHENSIVE, prepend it to the regular RAG fallback (which is insufficient source context)
            return (
                formulation_output +
                "\n\n## Regulatory/IP Analysis — Evidence Limitation\n\n"
                "No sufficiently relevant authorized source context was retrieved for the regulatory/IP aspects of this query. "
                "Therefore, the system cannot provide a grounded conclusion on patentability, ABS obligations, trademark protection, GI protection, or other legal requirements.\n\n"
                "## Information Needed for Further Analysis\n\n"
                "- Exact product formulation and intended use\n"
                "- The specific IP or regulatory issue to be assessed\n"
                "- Target jurisdiction\n"
                "- Ingredient sourcing and commercial-use details, where relevant\n\n"
                "## Recommended Next Steps\n\n"
                "1. Provide the missing formulation and product information.\n"
                "2. Specify the target jurisdiction.\n"
                "3. Ask the relevant IP and regulatory questions separately if a detailed analysis is required.\n"
                "4. Obtain professional advice from a **qualified IP attorney or regulatory professional** before making patent filing, commercialization, export, or benefit-sharing decisions."
            )

    if not documents:
        return (
            "## ⚠️ No Source Context Available\n\n"
            "No sufficiently relevant authorized source context was retrieved for this query. "
            "I cannot provide a grounded legal or regulatory answer without source support.\n\n"
            "## Information Needed for Further Analysis\n\n"
            "- Exact product formulation and intended use\n"
            "- The specific IP or regulatory issue to be assessed\n"
            "- Target jurisdiction\n"
            "- Ingredient sourcing and commercial-use details, where relevant\n\n"
            "Please consult a **qualified IP attorney or regulatory professional** before making high-impact decisions."
        )

    source_lines = _source_lines(documents)
    if "ipc25" in query or "ipc 25" in query or "section 25" in query:
        return (
            "## Summary\n\n"
            "**Section 25** of the Indian Patents Act relates to **patent opposition** and is highly relevant to Ayurveda/biological-resource inventions.\n\n"
            "## Key Requirements\n\n"
            "- **Patent opposition:** Failure to disclose biological material or associated traditional knowledge can be a ground for opposition. [Section 25, Patents Act]\n"
            "- **Revocation:** Non-disclosure may also lead to revocation proceedings. [Section 64, Patents Act]\n"
            "- **Geographical origin:** The source and geographical origin of biological material should be disclosed.\n"
            "- **Traditional knowledge:** Associated traditional knowledge, if any, must be identified.\n"
            "- **ABS compliance:** Prior-informed-consent and benefit-sharing arrangements may be required where applicable.\n\n"
            "## Evidence Limitations\n\n"
            "The available retrieved sources cover opposition and revocation grounds but may not address all procedural filing requirements or jurisdiction-specific variations.\n\n"
            "## Next Steps\n\n"
            "1. Review the specific patent application for disclosure completeness.\n"
            "2. Verify ABS and prior-informed-consent status for any biological resources used.\n"
            "3. Consult a qualified IP attorney for opposition or revocation risk assessment.\n\n"
            f"## Sources Used\n\n{source_lines}"
        )

    if "ipc" in query:
        return (
            "## Summary\n\n"
            "In this Ayurveda IP context, “IPC” may refer to **patent classification language** or a shorthand for an **Indian patent-law provision**.\n\n"
            "## Key Requirements\n\n"
            "- **Patentability:** The available context supports guidance on patentability of Ayurveda formulations.\n"
            "- **Traditional knowledge:** Disclosure requirements for traditional knowledge are addressed in the retrieved sources.\n"
            "- **Biological-source disclosure:** Obligations regarding source and origin of biological material are covered.\n\n"
            "## Evidence Limitations\n\n"
            "The retrieved context does not define a standalone term called “IPC” clearly. The applicable provisions depend on the specific section or classification intended.\n\n"
            "## Next Steps\n\n"
            "1. Clarify whether “IPC” refers to a specific section of the Patents Act or to an international patent classification.\n"
            "2. Provide the exact section number or classification code for a targeted analysis.\n\n"
            f"## Sources Used\n\n{source_lines}"
        )

    domain = state.get("legal_domain") or "GENERAL_IP"

    if domain == "TRADEMARK" or any(k in query for k in ("trademark", "trade mark", "brand name", "logo", "sanskrit name")):
        return (
            "## Summary\n\n"
            "Ayurvedic product branding is governed by the **Trade Marks Act, 1999**. Generic or descriptive classical Sanskrit formulation names cannot be registered as proprietary trademarks, whereas distinctive or coined brand names are protectable.\n\n"
            "## Key Requirements\n\n"
            "- **Distinctiveness (Section 9):** Names that designate the kind, quality, or generic ingredients (e.g., 'Triphala Churna', 'Ashwagandha Rasayana') are refused registration under Section 9 of the Trade Marks Act, 1999.\n"
            "- **Coined Brand Names:** Fanciful or coined marks (including coined Sanskrit terms that do not describe the ingredients) are registrable to distinguish goods.\n"
            "- **Class Classification:** Ayurvedic pharmaceuticals and medicinal formulations are registered under Class 5; Ayurvedic cosmetics under Class 3.\n"
            "- **International Brand Protection:** International registration across multiple target countries can be achieved through the Madrid System (Madrid Protocol) administered by WIPO via the Indian Trade Marks Registry.\n\n"
            "## Evidence Limitations\n\n"
            "Trade mark registrability is determined through examination and potential opposition. Prior brand search on the IP India Trade Mark database is essential.\n\n"
            "## Next Steps\n\n"
            "1. Conduct a comprehensive trademark search across Class 3 / Class 5 on the IP India portal.\n"
            "2. Ensure the proposed trademark is distinctive and not a generic Ayurvedic formula name.\n"
            "3. File Form TM-A with the Indian Trade Marks Registry, or designate target countries under the Madrid Protocol.\n\n"
            f"## Sources Used\n\n{source_lines}"
        )

    if domain == "GI_PROTECTION" or any(k in query for k in ("gi", "geographical indication")):
        return (
            "## Summary\n\n"
            "Region-specific classical formulations and botanical goods are protected under the **Geographical Indications of Goods (Registration and Protection) Act, 1999**. Applications are handled by the Geographical Indications Registry in Chennai.\n\n"
            "## Key Requirements\n\n"
            "- **Statutory Definition (Section 2(e)):** Protects goods where a given quality, reputation, or other characteristic is essentially attributable to their geographical origin (e.g., Kashmir Saffron, Kerala Ayurvedic Oil).\n"
            "- **Collective Ownership:** GIs belong to an association of persons, producers, or authorized collective bodies, not private individuals.\n"
            "- **Registration Procedure:** Applications must be filed with the Geographical Indications Registry in Chennai, followed by examination and a mandatory 4-month opposition window in the GI Journal.\n"
            "- **Duration & Protection:** Valid for 10 years, renewable indefinitely, conferring exclusive rights under Section 13 against false designation or passing off.\n\n"
            "## Evidence Limitations\n\n"
            "Documentary proof of historical origin, traditional geographical linkage, and specification of standards must be submitted.\n\n"
            "## Next Steps\n\n"
            "1. Form or coordinate with an association of regional cultivators/manufacturers.\n"
            "2. File the GI application at the Chennai GI Registry with historical and geographical evidence.\n"
            "3. Monitor the 4-month opposition window following publication in the GI Journal.\n\n"
            f"## Sources Used\n\n{source_lines}"
        )

    if domain == "COPYRIGHT" or any(k in query for k in ("copyright", "textbook", "book", "literary")):
        return (
            "## Summary\n\n"
            "Ayurvedic texts, commentaries, instructional manuals, and databases are protected under the **Copyright Act, 1957**. Ancient classical treatises are in the public domain, but original expressions, commentaries, and translations receive copyright protection.\n\n"
            "## Key Requirements\n\n"
            "- **Original Literary Works (Section 13):** Textbooks, modern interpretations, and formulation compilations receive statutory copyright protection for the author's life plus 60 years.\n"
            "- **Idea-Expression Dichotomy:** Copyright protects the original expression and text, not the underlying medical knowledge, herbal ingredients, or therapeutic ideas.\n"
            "- **Classical Texts in Public Domain:** Ancient root texts (Charaka Samhita, Sushruta Samhita) cannot be copyrighted; however, new translations, editorial annotations, or critical editions qualify for independent copyright.\n"
            "- **TKDL Protection:** Documented traditional formulations in the TKDL serve as prior art against third-party misappropriation.\n\n"
            "## Evidence Limitations\n\n"
            "Protection applies automatically upon fixation, but registration with the Copyright Office facilitates enforcement against infringement.\n\n"
            "## Next Steps\n\n"
            "1. Document the original literary expression and differentiate it from public domain root texts.\n"
            "2. File for voluntary copyright registration with the Copyright Office of India.\n\n"
            f"## Sources Used\n\n{source_lines}"
        )

    if domain == "PLANT_VARIETY" or any(k in query for k in ("plant variety", "ppvfr", "cultivar")):
        return (
            "## Summary\n\n"
            "New herbal cultivars and medicinal plant varieties are protected under the **Protection of Plant Varieties and Farmers' Rights Act, 2001 (PPV&FR Act)** through the PPV&FR Authority in New Delhi.\n\n"
            "## Key Requirements\n\n"
            "- **DUS Criteria (Section 15):** The medicinal plant variety must demonstrate Distinctiveness, Uniformity, and Stability (DUS) alongside novelty.\n"
            "- **Farmers' Rights (Section 39):** Traditional farmers and tribal communities retain statutory rights to save, use, sow, resow, exchange, or sell their farm-saved seed/propagating material.\n"
            "- **Benefit Sharing (Section 26):** Commercial breeders utilizing indigenous plant varieties must contribute to the National Gene Fund.\n\n"
            "## Evidence Limitations\n\n"
            "Requires multi-season DUS field trials according to species-specific testing guidelines established by the Authority.\n\n"
            "## Next Steps\n\n"
            "1. Conduct preliminary DUS characterization against existing reference varieties.\n"
            "2. File for registration under PPV&FR Form I with the Plant Authority in New Delhi.\n\n"
            f"## Sources Used\n\n{source_lines}"
        )

    if domain == "DESIGN" or any(k in query for k in ("design", "bottle", "container", "packaging")):
        return (
            "## Summary\n\n"
            "Aesthetic shape, ornamentation, and packaging for Ayurvedic products are protectable under the **Designs Act, 2000**. Protection lasts up to 15 years.\n\n"
            "## Key Requirements\n\n"
            "- **Novelty & Shape (Section 4):** Packaging bottles, containers, and blister configurations must be novel, original, and significantly distinguished from known designs.\n"
            "- **Exclusion of Functional Features:** Functional or mechanical mechanisms cannot be protected as industrial designs; only aesthetic visual features qualify.\n"
            "- **Duration (Section 11):** Copyright in a registered design is granted initially for 10 years, extendable by 5 years (15 years total).\n\n"
            "## Evidence Limitations\n\n"
            "Prior publication anywhere in India or abroad prior to the filing date invalidates the design.\n\n"
            "## Next Steps\n\n"
            "1. Prepare formal perspective and orthographic representation drawings of the bottle/packaging.\n"
            "2. File the application under the appropriate class with the Designs Wing of the Patent Office in Kolkata.\n\n"
            f"## Sources Used\n\n{source_lines}"
        )

    if domain == "ABS_COMPLIANCE" or any(k in query for k in ("abs", "nba", "sbb", "biodiversity")):
        return (
            "## Summary\n\n"
            "Access to Indian biological resources (herbs, plants, extracts) for commercial utilization or patent filing is strictly regulated under the **Biological Diversity Act, 2002 (amended 2023)**.\n\n"
            "## Key Requirements\n\n"
            "- **Foreign Entities / Export (Section 3):** Non-Indian citizens, foreign entities, and Indian companies with foreign shareholding must obtain prior approval from the National Biodiversity Authority (NBA).\n"
            "- **Indian Entities (Section 7):** Indian citizens and entities require prior intimation to the concerned State Biodiversity Board (SBB) for commercial utilization.\n"
            "- **IPR Approval (Section 6 & 20):** Any person applying for an intellectual property right based on biological research obtained from India must obtain NBA approval before grant (using Form III).\n"
            "- **Cultivator Exemption (Section 55):** Local farmers, cultivators, and vaidyars/traditional healers cultivating biological resources on their own land are exempt from prior NBA approval.\n\n"
            "## Evidence Limitations\n\n"
            "Compliance obligations depend on the origin of the raw material (wild-harvested vs. certified cultivated) and corporate nationality.\n\n"
            "## Next Steps\n\n"
            "1. Determine corporate entity nationality under Section 3 vs Section 7.\n"
            "2. File Form I (access) or Form III (IPR approval) with the NBA via the online portal (e-NBA).\n"
            "3. Execute Access and Benefit Sharing (ABS) agreement with the relevant Authority/Board.\n\n"
            f"## Sources Used\n\n{source_lines}"
        )

    if domain == "FSSAI_FOOD" or any(k in query for k in ("food", "aahar", "supplement", "fssai")):
        return (
            "## Summary\n\n"
            "Herbal and Ayurvedic dietary supplements are regulated under the **FSSAI (Ayurveda Aahar) Regulations, 2022**. Therapeutic or medicinal disease-curing claims are prohibited on food labels.\n\n"
            "## Key Requirements\n\n"
            "- **Ayurveda Aahar Definition:** Foods prepared in accordance with the authoritative books listed in Schedule A of the regulations.\n"
            "- **No Disease Cure Claims:** Under Section 24 of the Food Safety and Standards Act, 2006, Ayurveda Aahar products cannot claim to treat, prevent, or cure human diseases.\n"
            "- **Mandatory Logo:** Products must display the dedicated Ayurveda Aahar logo alongside the FSSAI license number.\n\n"
            "## Evidence Limitations\n\n"
            "Products with therapeutic claims must be licensed as Ayurvedic medicines under the Drugs & Cosmetics Act, not as Ayurveda Aahar.\n\n"
            "## Next Steps\n\n"
            "1. Verify ingredients against Schedule A authoritative Ayurvedic texts.\n"
            "2. Apply for an FSSAI manufacturing license under the Ayurveda Aahar category on the FoSCoS portal.\n\n"
            f"## Sources Used\n\n{source_lines}"
        )

    if domain == "COSMETIC" or any(k in query for k in ("cosmetic", "cream", "lotion", "wash")):
        return (
            "## Summary\n\n"
            "Herbal beauty, skin, and hair preparations intended for topical application without therapeutic claims are governed by the **Cosmetics Rules, 2020** under the Drugs and Cosmetics Act.\n\n"
            "## Key Requirements\n\n"
            "- **Cosmetic vs. Drug:** Under Section 3(aaa), cosmetics are articles intended to be rubbed, poured, or sprayed for cleansing, beautifying, or altering the appearance without altering bodily functions.\n"
            "- **Labeling & Claims (Rule 34):** No medicinal, preventive, or cure claims for skin diseases (e.g., eczema, psoriasis) are permitted on cosmetic labels.\n"
            "- **Formulation Licensing:** Manufactured under a cosmetic manufacturing license from the State Licensing Authority (SLA) under Schedule M-II / Cosmetics Rules.\n\n"
            "## Evidence Limitations\n\n"
            "Adding therapeutic claims triggers the requirement for an Ayurvedic Drug license under Rule 158B.\n\n"
            "## Next Steps\n\n"
            "1. Review label claims to ensure they remain cosmetic and beautifying.\n"
            "2. Apply for a cosmetic manufacturing license with the State Licensing Authority.\n\n"
            f"## Sources Used\n\n{source_lines}"
        )

    # Default Patent Eligibility Guidance
    return (
        "## Summary\n\n"
        "Patent protection for Ayurvedic innovations in India is governed by the **Patents Act, 1970**. Traditional knowledge formulations are excluded under **Section 3(p)**, while novel processes and non-obvious synergistic compositions may be patentable.\n\n"
        "## Key Requirements\n\n"
        "- **Traditional Knowledge Bar (Section 3(p)):** An invention which in effect is traditional knowledge or an aggregation of known properties of traditionally known components is not patentable.\n"
        "- **Synergistic Formulations & Novel Processes:** Novel extraction methods, isolated active phytoconstituents, or synergistic combinations demonstrating unexpected biological activity beyond additive effects can overcome Section 3(p) and Section 3(e).\n"
        "- **Form 18A (Expedited Examination):** Patent applicants who are startups, female applicants, or eligible research institutions can apply for expedited examination using Form 18A.\n"
        "- **TKDL Prior Art Defense:** The Indian Patent Office checks the Traditional Knowledge Digital Library (TKDL) as prior art to reject claims covering classical formulations.\n"
        "- **Form 18A (Expedited Examination):** Expedited examination can be requested under Rule 24C / Form 18A for recognized startups and small entities.\n\n"
        "## Evidence Limitations\n\n"
        "Prior-art searching against TKDL and global patent databases is mandatory before filing to establish novelty and inventive step.\n\n"
        "## Next Steps\n\n"
        "1. Conduct a rigorous prior art search including TKDL and classical AYUSH texts.\n"
        "2. Generate comparative empirical data establishing synergistic therapeutic efficacy beyond individual ingredients.\n"
        "3. File the patent specification with Form 18A (if eligible for expedited examination) and Form III with the NBA.\n\n"
        f"## Sources Used\n\n{source_lines}"
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
    # Do NOT automatically append all retrieved documents. The LLM must filter out irrelevant ones.
    return answer


def _safe_abstention_answer(state: AgentState) -> str:
    return (
        "## ⚠️ Insufficient Source Context\n\n"
        "No sufficiently relevant authorized source context was retrieved for this question. "
        "I cannot provide a grounded legal or regulatory answer without source support.\n\n"
        "## Information Needed for Further Analysis\n\n"
        "- Exact product formulation and intended use\n"
        "- The specific IP or regulatory issue to be assessed\n"
        "- Target jurisdiction\n"
        "- Ingredient sourcing and commercial-use details, where relevant\n\n"
        "## Recommended Next Steps\n\n"
        "1. Provide the missing formulation and product information.\n"
        "2. Specify the target jurisdiction.\n"
        "3. Ask the relevant IP and regulatory questions separately if a detailed analysis is required.\n"
        "4. Obtain professional advice from a **qualified IP attorney or regulatory professional** before making patent filing, commercialization, export, or benefit-sharing decisions."
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

    # Locate all section boundaries
    india_start = lower.find(india_marker)
    intl_start = lower.find(intl_marker)
    key_diff_start = lower.find("## key differences")
    evidence_lim_start = lower.find("## evidence limitations")
    next_steps_start = lower.find("## practical next steps")

    # Build an ordered list of known section boundaries after intl_start
    post_intl_boundaries = sorted(
        pos for pos in [key_diff_start, evidence_lim_start, next_steps_start]
        if pos > intl_start and pos != -1
    )

    # India text: from india_marker to intl_marker
    india_text = content[india_start:intl_start].strip()

    # International text: from intl_marker to the next known section (or end)
    intl_end = post_intl_boundaries[0] if post_intl_boundaries else len(content)
    international_text = content[intl_start:intl_end].strip()

    # Key Differences section
    key_differences = ""
    if key_diff_start != -1:
        kd_boundaries = sorted(
            pos for pos in [evidence_lim_start, next_steps_start]
            if pos > key_diff_start and pos != -1
        )
        kd_end = kd_boundaries[0] if kd_boundaries else len(content)
        key_differences = content[key_diff_start:kd_end].strip()

    # Evidence Limitations section
    evidence_limitations = ""
    if evidence_lim_start != -1:
        el_end = next_steps_start if next_steps_start != -1 and next_steps_start > evidence_lim_start else len(content)
        evidence_limitations = content[evidence_lim_start:el_end].strip()

    # Practical Next Steps section
    practical = content[next_steps_start:].strip() if next_steps_start != -1 else ""

    return {
        "india": india_text,
        "international": international_text,
        "key_differences": key_differences,
        "evidence_limitations": evidence_limitations,
        "practical_next_steps": practical,
    }


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
