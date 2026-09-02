# ============================================================
# IP-SAKTI Sahayak — FastAPI Entry Point
# Ayurveda IP & Regulatory Guidance Assistant
# ============================================================
from app.observability.logfire_compat import logfire
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Callable

from dotenv import load_dotenv

load_dotenv()
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))

from fastapi import FastAPI, Response, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List

from app.agents.graph import rag_agent
from app.agents.nodes.formulation_classifier import formulation_classifier_node
from app.guardrails import initialize_rails, guard
from app.services.multilingual import (
    detect_language,
    translate_to_english,
    translate_from_english,
    get_supported_languages,
)
from app.observability.instrumentation import (
    log_api_request, log_api_response, emit_query_event,
    APIRequest, APIResponse, QueryEvent,
    IntentType, JurisdictionType, FormulationType, GuardrailAction
)


# ============================================================
# LIFESPAN & STARTUP
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_rails()
    # Build in-memory BM25 keyword index from Qdrant corpus
    from app.services.retrieval.qdrant_service import get_all_documents
    from app.services.retrieval.bm25_service import build_bm25_index
    docs = get_all_documents()
    if docs:
        build_bm25_index(docs)
    logfire.info("🚀 IP-SAKTI Sahayak API started")
    yield
    logfire.info("🛑 IP-SAKTI Sahayak API shutting down")


app = FastAPI(
    title="IP-SAKTI Sahayak API",
    description="Ayurveda IP & Regulatory Guidance Assistant",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:8501,http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

REQUEST_COUNTS = {}
RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))


async def rate_limit(request: Request):
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    minute_key = int(now // 60)
    key = f"{client_ip}:{minute_key}"
    
    REQUEST_COUNTS[key] = REQUEST_COUNTS.get(key, 0) + 1
    if REQUEST_COUNTS[key] > RATE_LIMIT:
        logfire.warning(f"Rate limit exceeded: {client_ip}")
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")


API_KEY = os.getenv("API_KEY")


async def verify_api_key(request: Request):
    if API_KEY:
        provided = request.headers.get("X-API-Key") or request.query_params.get("api_key")
        if provided != API_KEY:
            logfire.warning(f"Invalid API key from {request.client.host}")
            raise HTTPException(status_code=401, detail="Invalid API key")


# ============================================================
# REQUEST/RESPONSE MODELS
# ============================================================

class QueryRequest(BaseModel):
    q: str = Field(..., min_length=1, max_length=2000, description="User query")
    thread_id: Optional[str] = Field(default="default_user", max_length=100)
    language: Optional[str] = Field(default="en", max_length=10, description="Response language code (e.g., 'hi', 'ta')")
    jurisdiction: Optional[str] = Field(default=None, max_length=20, description="Optional selected scope: INDIA, INTERNATIONAL, or BOTH")


class ClassificationRequest(BaseModel):
    description: str = Field(..., min_length=1, max_length=1000)
    ingredients: Optional[str] = Field(default=None, max_length=500)
    intended_use: Optional[str] = Field(default=None, max_length=200)
    reference_to_classical_text: Optional[str] = Field(default=None, max_length=500)
    language: Optional[str] = Field(default="en", max_length=10)


class SourceCitation(BaseModel):
    text: str
    source: Optional[str] = None
    section: Optional[str] = None
    url: Optional[str] = None
    verified: Optional[bool] = None
    confidence: Optional[float] = Field(default=None, ge=0, le=1)


class ABSHelper(BaseModel):
    required_or_possible: bool = False
    status: str = "not_clearly_indicated"
    next_steps: List[str] = []


class TKDLPriorArtPointer(BaseModel):
    relevant: bool = False
    pointer: str = ""


class EscalationPath(BaseModel):
    recommended: bool = False
    path: str = ""
    reasons: List[str] = []


class QueryResponse(BaseModel):
    question: str
    answer: str
    intent: Optional[str] = None
    jurisdiction: Optional[str] = None
    formulation_type: Optional[str] = None
    citations: List[SourceCitation] = []
    confidence_score: Optional[float] = Field(default=None, ge=0, le=1)
    confidence_label: Optional[str] = None  # High / Medium / Low — authoritative display label
    abs_required: Optional[bool] = None
    abs_helper: Optional[ABSHelper] = None
    tkdl_prior_art_pointer: Optional[TKDLPriorArtPointer] = None
    jurisdiction_answer_sets: Optional[dict] = None
    escalation: Optional[EscalationPath] = None
    thought_process: List[str] = []
    status: str = "complete"
    sources: List[str] = []
    chunks_retrieved: int = 0
    language: str = "en"
    original_language: Optional[str] = None
    disclaimer: str = "Information only, not legal advice."


class ErrorResponse(BaseModel):
    error: str
    error_code: str
    request_id: str
    detail: Optional[str] = None


# ============================================================
# EXCEPTION HANDLERS
# ============================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    request_id = str(uuid.uuid4())[:8]
    logfire.error(f"HTTP {exc.status_code}: {exc.detail} | req={request_id}")
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            error_code=f"HTTP_{exc.status_code}",
            request_id=request_id
        ).model_dump()
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    request_id = str(uuid.uuid4())[:8]
    logfire.exception(f"Unhandled error: {exc} | req={request_id}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            error_code="INTERNAL_ERROR",
            request_id=request_id,
            detail=str(exc) if os.getenv("DEBUG") == "true" else None
        ).model_dump()
    )


# ============================================================
# HEALTH & READINESS
# ============================================================

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "ip-sakti-sahayak"}


@app.get("/ready")
async def readiness_check():
    from app.services.retrieval.qdrant_service import client
    from app.services.retrieval.embedding import embed_query
    from app.gateway.client import portkey_client
    from app.config import settings
    
    checks = {}
    
    try:
        if client is None:
            checks["qdrant"] = "degraded: qdrant-client not installed"
        else:
            checks["qdrant"] = "ok" if client.collection_exists("enterprise_rag") else "degraded"
    except Exception as e:
        checks["qdrant"] = f"error: {e}"
    
    try:
        _ = embed_query("test")
        checks["embeddings"] = "ok"
    except Exception as e:
        checks["embeddings"] = f"error: {e}"
    
    try:
        _ = portkey_client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1
        )
        checks["llm_gateway"] = "ok"
    except Exception as e:
        checks["llm_gateway"] = f"error: {e}"
    
    all_ok = all(v == "ok" for v in checks.values())
    status_code = 200 if all_ok else 503
    
    return JSONResponse(
        status_code=status_code,
        content={"status": "ready" if all_ok else "degraded", "checks": checks}
    )


# ============================================================
# CORE ENDPOINTS
# ============================================================

@app.get("/")
def home():
    return {
        "message": "IP-SAKTI Sahayak API is live.",
        "scope": "Ayurveda IP & Regulatory Guidance",
        "disclaimer": "This is informational only, not legal advice.",
        "docs": "/docs",
        "multilingual": "/languages"
    }


@app.get("/languages")
def list_languages():
    """Return all supported languages for multilingual queries."""
    return {
        "supported_languages": get_supported_languages(),
        "default": "en",
        "note": "Set 'language' parameter in /query or /classify to receive responses in that language."
    }


@app.get("/graph")
def get_graph_image():
    try:
        png_bytes = rag_agent.get_graph().draw_mermaid_png()
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        logfire.error(f"Graph image generation failed: {e}")
        return {"error": f"Could not generate graph image: {e}"}


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(rate_limit), Depends(verify_api_key)])
def query(request: QueryRequest, http_request: Request):
    start_total = time.perf_counter()
    request_id = str(uuid.uuid4())[:8]
    thread_id = request.thread_id or "default_user"
    target_language = request.language or "en"
    requested_jurisdiction = (request.jurisdiction or "").upper() or None
    if requested_jurisdiction and requested_jurisdiction not in {"INDIA", "INTERNATIONAL", "BOTH"}:
        raise HTTPException(status_code=422, detail="jurisdiction must be INDIA, INTERNATIONAL, or BOTH")
    
    # Language detection and translation
    original_language = detect_language(request.q)
    query_text = request.q
    
    if original_language != "en":
        translation_result = translate_to_english(request.q, original_language)
        query_text = translation_result["text"]
        logfire.info(f"Translated query from {original_language} to en: {query_text[:100]}...")
    
    log_api_request(APIRequest(
        method="POST", path="/query", thread_id=thread_id, query_length=len(request.q)
    ))

    initial_state = {
        "messages": [{"role": "user", "content": query_text}],
        "current_query": query_text,
        "documents": [],
        "plan": ["Start"],
        "status": "Initializing IP-SAKTI pipeline...",
        "intent": None,
        "jurisdiction": None,
        "requested_jurisdiction": requested_jurisdiction,
        "formulation_type": None,
        "citations": [],
        "confidence_score": None,
        "abs_required": None,
        "abs_helper": None,
        "tkdl_prior_art_pointer": None,
        "jurisdiction_answer_sets": None,
        "escalation": None,
        "refusal_reason": None,
        "thread_id": thread_id,
        "request_id": request_id
    }

    config = {"configurable": {"thread_id": thread_id}}

    try:
        guard_start = time.perf_counter()
        rail_fired, rail_response = guard(query_text)
        guard_latency = int((time.perf_counter() - guard_start) * 1000)

        if rail_fired:
            logfire.info(f"🛡️ Blocked by guardrails | thread={thread_id} req={request_id}")
            
            # Translate response if needed
            if target_language != "en":
                translated = translate_from_english(rail_response, target_language)
                rail_response = translated["text"]
            
            resp = QueryResponse(
                question=request.q,
                answer=rail_response,
                intent="BLOCKED",
                thought_process=["Intent: Guardrails Fired"],
                status="Blocked by guardrails.",
                sources=[],
                chunks_retrieved=0,
                language=target_language,
                original_language=original_language if original_language != "en" else None,
                escalation=EscalationPath(
                    recommended=False,
                    path="No escalation path is needed for blocked unsafe or off-scope requests.",
                    reasons=["The request was blocked by guardrails."]
                )
            )
            return resp

        rag_start = time.perf_counter()
        final_output = rag_agent.invoke(initial_state, config=config)
        rag_latency = int((time.perf_counter() - rag_start) * 1000)

        total_latency = int((time.perf_counter() - start_total) * 1000)

        answer_text = final_output.get("final_answer", "")
        
        # Translate answer to target language if needed
        if target_language != "en":
            translated = translate_from_english(answer_text, target_language)
            answer_text = translated["text"]
            logfire.info(f"Translated answer to {target_language}")

        sanitized = _sanitize_metadata(final_output)

        emit_query_event(QueryEvent(
            thread_id=thread_id,
            user_query=query_text,
            intent=_enum_or_unknown(IntentType, final_output.get("intent")),
            jurisdiction=_enum_or_unknown(JurisdictionType, final_output.get("jurisdiction")),
            formulation_type=_optional_enum(FormulationType, final_output.get("formulation_type")),
            guardrail_action=GuardrailAction.PASSED,
            retrieval_latency_ms=rag_latency,
            llm_latency_ms=0,
            total_latency_ms=total_latency,
            citations_count=len(final_output.get("citations", [])),
            confidence=sanitized["confidence_score"],
            abs_required=final_output.get("abs_required"),
            status="SUCCESS"
        ))

        resp = QueryResponse(
            question=request.q,
            answer=answer_text,
            intent=final_output.get("intent"),
            jurisdiction=final_output.get("jurisdiction"),
            formulation_type=sanitized["formulation_type"],
            citations=final_output.get("citations", []),
            confidence_score=sanitized["confidence_score"],
            abs_required=sanitized["abs_required"],
            abs_helper=sanitized["abs_helper"],
            tkdl_prior_art_pointer=sanitized["tkdl_prior_art_pointer"],
            jurisdiction_answer_sets=final_output.get("jurisdiction_answer_sets"),
            escalation=final_output.get("escalation"),
            thought_process=final_output.get("plan", []),
            status=final_output.get("status", "complete"),
            sources=final_output.get("documents", []),
            chunks_retrieved=len(final_output.get("documents", [])),
            language=target_language,
            original_language=original_language if original_language != "en" else None,
            confidence_label=sanitized["confidence_label"],
        )

        return resp

    except Exception as e:
        total_latency = int((time.perf_counter() - start_total) * 1000)
        logfire.error(f"❌ Backend Execution Failed: {e} | req={request_id}")
        
        return QueryResponse(
            question=request.q,
            answer="I apologize, but I encountered an internal error. Please try again later.",
            thought_process=["Error encountered during execution."],
            status="error",
            sources=[],
            chunks_retrieved=0,
            language=target_language
        )


def _enum_or_unknown(enum_cls, value):
    try:
        return enum_cls(value or "UNKNOWN")
    except Exception:
        return enum_cls("UNKNOWN")


def _optional_enum(enum_cls, value):
    if not value:
        return None
    try:
        return enum_cls(value)
    except Exception:
        return None


def _sanitize_metadata(final_output: dict) -> dict:
    """
    Normalizes formulation_type, abs_required, and confidence_score so that
    the API trust metadata can never contradict the grounded answer.

    Rules:
    - confidence >= 0.70  → High    → show raw formulation_type; show abs_required as-is
    - 0.50 <= confidence < 0.70 → Medium → show formulation_type; show abs_required as 'Assessment Recommended'
    - confidence < 0.50 or INSUFFICIENT_INFO → Low → suppress formulation_type; suppress abs
    - No citations → cap display confidence label at Medium regardless of score
    """
    confidence = final_output.get("confidence_score")
    formulation_type = final_output.get("formulation_type") or "INSUFFICIENT_INFO"
    abs_required = final_output.get("abs_required")
    citations = final_output.get("citations") or []

    # Determine categorical confidence label
    if confidence is None or formulation_type == "INSUFFICIENT_INFO":
        conf_label = "Low"
        conf_value = min(confidence or 0.0, 0.45)  # cap numeric at <50%
    elif confidence >= 0.70:
        conf_label = "High"
        conf_value = confidence
    elif confidence >= 0.50:
        conf_label = "Medium"
        conf_value = confidence
    else:
        conf_label = "Low"
        conf_value = confidence

    # If no claim-level citations, cap at Medium
    if not citations and conf_label == "High":
        conf_label = "Medium"

    # Suppress definitive formulation_type when confidence is Low
    if conf_label == "Low" or formulation_type == "INSUFFICIENT_INFO":
        safe_formulation_type = "UNDETERMINED"
    else:
        safe_formulation_type = formulation_type

    # Normalize abs_required display value based on confidence level
    if conf_label == "High" and abs_required is True:
        abs_display = True          # confirmed applicable
    elif conf_label == "High" and abs_required is False:
        abs_display = False         # confirmed not required
    elif conf_label == "Medium" or (conf_label == "High" and abs_required is None):
        abs_display = None          # assessment recommended
    else:  # Low
        abs_display = None          # cannot determine

    # Suppress ABS helper content unless citations support it
    abs_helper = final_output.get("abs_helper")
    if not citations and abs_helper:
        abs_helper = {
            "status": "insufficient_evidence",
            "next_steps": ["Insufficient authorized source context to determine applicable ABS requirements. Further assessment is required."]
        }

    # Suppress TKDL pointer unless citations support it
    tkdl_pointer = final_output.get("tkdl_prior_art_pointer")
    if not citations and tkdl_pointer:
        tkdl_pointer = {
            "relevant": tkdl_pointer.get("relevant", False),
            "pointer": "Insufficient authorized source context to determine TKDL/prior-art applicability. Formulation-specific evidence is required."
        }

    return {
        "formulation_type": safe_formulation_type,
        "abs_required": abs_display,
        "confidence_score": conf_value,
        "confidence_label": conf_label,
        "abs_helper": abs_helper,
        "tkdl_prior_art_pointer": tkdl_pointer,
    }


@app.post("/classify", dependencies=[Depends(rate_limit), Depends(verify_api_key)])
def classify_formulation(request: ClassificationRequest, http_request: Request):
    start = time.perf_counter()
    request_id = str(uuid.uuid4())[:8]
    thread_id = "classify"
    target_language = request.language or "en"
    
    combined = f"{request.description}"
    if request.ingredients:
        combined += f"\nIngredients: {request.ingredients}"
    if request.intended_use:
        combined += f"\nIntended use: {request.intended_use}"
    if request.reference_to_classical_text:
        combined += f"\nClassical text reference: {request.reference_to_classical_text}"

    initial_state = {
        "messages": [{"role": "user", "content": combined}],
        "current_query": combined,
        "documents": [],
        "plan": ["Quick classification"],
        "status": "Classifying formulation...",
        "intent": "FORMULATION_CLASSIFICATION",
        "jurisdiction": "INDIA",
        "requested_jurisdiction": "INDIA",
        "formulation_type": None,
        "citations": [],
        "confidence_score": None,
        "abs_required": None,
        "abs_helper": None,
        "tkdl_prior_art_pointer": None,
        "jurisdiction_answer_sets": None,
        "escalation": None,
        "refusal_reason": None,
        "thread_id": thread_id,
        "request_id": request_id
    }

    try:
        result = {
            **initial_state,
            **formulation_classifier_node(initial_state),
        }
        latency_ms = int((time.perf_counter() - start) * 1000)

        formulation_type = result.get("formulation_type") or "INSUFFICIENT_INFO"
        abs_required = result.get("abs_required")
        confidence_score = result.get("confidence_score")
        if confidence_score is None:
            confidence_score = 0.45 if formulation_type == "INSUFFICIENT_INFO" else 0.55

        explanation = _classification_explanation(formulation_type, abs_required, confidence_score)
        if target_language != "en":
            translated = translate_from_english(explanation, target_language)
            explanation = translated["text"]

        return {
            "formulation_type": formulation_type,
            "abs_required": abs_required,
            "confidence_score": confidence_score,
            "explanation": explanation,
            "latency_ms": latency_ms,
            "language": target_language
        }
    except Exception as e:
        logfire.error(f"Classification failed: {e} | req={request_id}")
        return {
            "error": str(e),
            "formulation_type": "INSUFFICIENT_INFO",
            "abs_required": None,
            "confidence_score": 0.0,
            "explanation": "Classification could not be completed. Please add ingredients, intended use, claims, and any classical-text reference.",
            "language": target_language
        }


def _classification_explanation(formulation_type: str, abs_required: Optional[bool], confidence_score: float) -> str:
    guidance = {
        "CLASSICAL_AYURVEDIC": "The product appears to follow a classical Ayurvedic pathway because a classical text reference or First Schedule signal is present.",
        "PROPRIETARY_AYURVEDIC": "The product appears to be a proprietary Ayurvedic medicine because it is a therapeutic herbal product without a clear classical text reference.",
        "PHYTOPHARMACEUTICAL": "The product appears closer to the phytopharmaceutical pathway and would need stronger standardization and evidence.",
        "FOOD_AYURVEDA_AAHAR": "The product appears closer to a food or nutraceutical pathway such as Ayurveda Aahar.",
        "COSMETIC": "The product appears closer to a cosmetic or personal-care pathway, where therapeutic disease claims should be avoided.",
        "INSUFFICIENT_INFO": "There is not enough detail to confidently classify the product pathway.",
    }
    abs_text = "may be required" if abs_required else "is not clearly indicated from the provided details"
    return (
        f"{guidance.get(formulation_type, guidance['INSUFFICIENT_INFO'])} "
        f"ABS/biodiversity compliance {abs_text}. "
        f"Confidence is {confidence_score:.0%}; add exact composition, claims, manufacturing process, and any classical-text reference to refine this."
    )


@app.get("/jurisdictions")
def list_jurisdictions():
    return {
        "jurisdictions": [
            {"code": "INDIA", "name": "India", "description": "Indian patents, GI, trademarks, AYUSH, FSSAI, BD Act"},
            {"code": "INTERNATIONAL", "name": "International", "description": "WIPO, PCT, TRIPS, Nagoya Protocol"},
            {"code": "BOTH", "name": "Both", "description": "Comparative or cross-jurisdictional queries"}
        ]
    }


@app.get("/formulation-types")
def list_formulation_types():
    return {
        "types": [
            {"code": "CLASSICAL_AYURVEDIC", "name": "Classical Ayurvedic", "pathway": "Drugs & Cosmetics Act First Schedule"},
            {"code": "PROPRIETARY_AYURVEDIC", "name": "Patent/Proprietary Ayurvedic", "pathway": "AYUSH licensing, possible patent"},
            {"code": "PHYTOPHARMACEUTICAL", "name": "Phytopharmaceutical", "pathway": "D&C Rules phytopharmaceutical pathway"},
            {"code": "FOOD_AYURVEDA_AAHAR", "name": "Ayurveda Aahar (Food)", "pathway": "FSSAI regulations"},
            {"code": "COSMETIC", "name": "Cosmetic", "pathway": "Cosmetic Rules, no therapeutic claims"}
        ]
    }
