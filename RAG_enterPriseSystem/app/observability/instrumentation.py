"""
Logfire instrumentation helpers for IP-SAKTI nodes.
Each node gets structured spans with Pydantic models.
"""

from app.observability.logfire_compat import logfire
import time
from typing import Optional, List, Dict, Any
from functools import wraps

from app.observability.schemas import (
    PlannerInput, PlannerOutput,
    JurisdictionRouterInput, JurisdictionRouterOutput,
    FormulationClassifierInput, FormulationClassifierOutput,
    RetrievalInput, RetrievalOutput, RetrievedChunk,
    LLMGenerationInput, LLMGenerationOutput,
    CitationVerificationInput, CitationVerificationOutput,
    GuardrailInput, GuardrailOutput,
    APIRequest, APIResponse,
    QueryEvent, ClassificationEvent,
    IntentType, JurisdictionType, FormulationType, GuardrailAction, NodeStatus
)
from app.security.privacy import redact_personal_data


# ============================================================
# SPAN DECORATORS
# ============================================================

def span_planner(func):
    @wraps(func)
    def wrapper(state, *args, **kwargs):
        query = state.get("current_query", "")
        messages = state.get("messages", [])
        thread_id = state.get("thread_id", "unknown")
        
        with logfire.span("planner_node", 
                          user_query=redact_personal_data(query),
                          conversation_length=len(messages),
                          thread_id=thread_id) as span:
            start = time.perf_counter()
            result = func(state, *args, **kwargs)
            latency_ms = int((time.perf_counter() - start) * 1000)
            
            intent = result.get("intent", "UNKNOWN")
            span.set_attribute("intent", intent)
            span.set_attribute("latency_ms", latency_ms)
            logfire.info(f"Planner: intent={intent}, latency={latency_ms}ms")
            return result
    return wrapper


def span_jurisdiction_router(func):
    @wraps(func)
    def wrapper(state, *args, **kwargs):
        query = state.get("current_query", "")
        intent = state.get("intent", "")
        
        with logfire.span("jurisdiction_router",
                          query=redact_personal_data(query),
                          intent=intent) as span:
            start = time.perf_counter()
            result = func(state, *args, **kwargs)
            latency_ms = int((time.perf_counter() - start) * 1000)
            
            jurisdiction = result.get("jurisdiction", "UNKNOWN")
            span.set_attribute("jurisdiction", jurisdiction)
            span.set_attribute("latency_ms", latency_ms)
            logfire.info(f"Jurisdiction Router: {jurisdiction}, latency={latency_ms}ms")
            return result
    return wrapper


def span_formulation_classifier(func):
    @wraps(func)
    def wrapper(state, *args, **kwargs):
        query = state.get("current_query", "")
        jurisdiction = state.get("jurisdiction", "")
        
        with logfire.span("formulation_classifier",
                          query=redact_personal_data(query),
                          jurisdiction=jurisdiction) as span:
            start = time.perf_counter()
            result = func(state, *args, **kwargs)
            latency_ms = int((time.perf_counter() - start) * 1000)
            
            ftype = result.get("formulation_type", "UNKNOWN")
            abs_req = result.get("abs_required")
            confidence = result.get("confidence_score", 0)
            span.set_attribute("formulation_type", ftype)
            span.set_attribute("abs_required", str(abs_req))
            span.set_attribute("confidence", confidence)
            span.set_attribute("latency_ms", latency_ms)
            logfire.info(f"Classifier: {ftype}, ABS={abs_req}, conf={confidence:.2f}, latency={latency_ms}ms")
            return result
    return wrapper


def span_retrieval(func):
    @wraps(func)
    def wrapper(state, *args, **kwargs):
        query = state.get("current_query", "")
        jurisdiction = state.get("jurisdiction", "")
        ftype = state.get("formulation_type")
        
        with logfire.span("retrieval_node",
                          query=redact_personal_data(query),
                          jurisdiction=jurisdiction,
                          formulation_type=ftype or "NONE") as span:
            start = time.perf_counter()
            result = func(state, *args, **kwargs)
            latency_ms = int((time.perf_counter() - start) * 1000)
            
            docs = result.get("documents", [])
            span.set_attribute("chunks_retrieved", len(docs))
            span.set_attribute("latency_ms", latency_ms)
            logfire.info(f"Retrieval: {len(docs)} chunks, latency={latency_ms}ms")
            return result
    return wrapper


def span_responder(func):
    @wraps(func)
    def wrapper(state, *args, **kwargs):
        intent = state.get("intent", "")
        jurisdiction = state.get("jurisdiction", "")
        ftype = state.get("formulation_type")
        doc_count = len(state.get("documents", []))
        
        with logfire.span("responder_node",
                          intent=intent,
                          jurisdiction=jurisdiction,
                          formulation_type=ftype or "NONE",
                          context_chunks=doc_count) as span:
            start = time.perf_counter()
            result = func(state, *args, **kwargs)
            latency_ms = int((time.perf_counter() - start) * 1000)
            
            answer = result.get("final_answer", "")
            span.set_attribute("answer_length", len(answer))
            span.set_attribute("latency_ms", latency_ms)
            span.set_attribute("cache_status", "UNKNOWN")  # Portkey sets this
            logfire.info(f"Responder: {len(answer)} chars, latency={latency_ms}ms")
            return result
    return wrapper


def span_citation_verifier(func):
    @wraps(func)
    def wrapper(state, *args, **kwargs):
        answer = state.get("final_answer", "")
        docs = state.get("documents", [])
        
        with logfire.span("citation_verifier",
                          answer_length=len(answer),
                          context_chunks=len(docs)) as span:
            start = time.perf_counter()
            result = func(state, *args, **kwargs)
            latency_ms = int((time.perf_counter() - start) * 1000)
            
            citations = result.get("citations", [])
            confidence = result.get("confidence_score", 0)
            span.set_attribute("citations_found", len(citations))
            span.set_attribute("confidence", confidence)
            span.set_attribute("latency_ms", latency_ms)
            logfire.info(f"Citation Verifier: {len(citations)} citations, conf={confidence:.2f}, latency={latency_ms}ms")
            return result
    return wrapper


# ============================================================
# GUARDRAIL SPAN
# ============================================================

def log_guardrail_check(query: str, thread_id: str, action: GuardrailAction, 
                        response: Optional[str], indicators: List[str], latency_ms: int):
    """Log guardrail check as structured event."""
    logfire.info("Guardrail Check",
                 user_query=redact_personal_data(query)[:200],
                 thread_id=thread_id,
                 action=action.value,
                 indicators_matched=indicators,
                 latency_ms=latency_ms,
                 blocked=action != GuardrailAction.PASSED)


# ============================================================
# API REQUEST/RESPONSE LOGGING
# ============================================================

def log_api_request(request: APIRequest):
    logfire.info("API Request",
                 method=request.method,
                 path=request.path,
                 thread_id=request.thread_id,
                 query_length=request.query_length)


def log_api_response(response: APIResponse):
    logfire.info("API Response",
                 status_code=response.status_code,
                 thread_id=response.thread_id,
                 intent=response.intent.value if response.intent else None,
                 jurisdiction=response.jurisdiction.value if response.jurisdiction else None,
                 formulation_type=response.formulation_type.value if response.formulation_type else None,
                 confidence=response.confidence,
                 abs_required=response.abs_required,
                 citations_count=response.citations_count,
                 latency_ms=response.latency_ms,
                 error=response.error)


def emit_query_event(event: QueryEvent):
    """Emit top-level query event for dashboard."""
    logfire.info("Query Event",
                 timestamp=event.timestamp.isoformat(),
                 thread_id=event.thread_id,
                 user_query=redact_personal_data(event.user_query)[:200],
                 intent=event.intent.value,
                 jurisdiction=event.jurisdiction.value,
                 formulation_type=event.formulation_type.value if event.formulation_type else None,
                 guardrail_action=event.guardrail_action.value,
                 retrieval_latency_ms=event.retrieval_latency_ms,
                 llm_latency_ms=event.llm_latency_ms,
                 total_latency_ms=event.total_latency_ms,
                 citations_count=event.citations_count,
                 confidence=event.confidence,
                 abs_required=event.abs_required,
                 status=event.status,
                 error=event.error)


def emit_classification_event(event: ClassificationEvent):
    """Emit classification event for dashboard."""
    logfire.info("Classification Event",
                 timestamp=event.timestamp.isoformat(),
                 thread_id=event.thread_id,
                 description=redact_personal_data(event.description)[:200],
                 formulation_type=event.formulation_type.value,
                 abs_required=event.abs_required,
                 confidence=event.confidence,
                 latency_ms=event.latency_ms)
