"""
Pydantic schemas for structured Logfire observability.
All spans use these models for consistent, queryable telemetry.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from enum import Enum


# ============================================================
# ENUMS
# ============================================================

class IntentType(str, Enum):
    REGULATORY_IP = "REGULATORY_IP"
    FORMULATION_CLASSIFICATION = "FORMULATION_CLASSIFICATION"
    CONVERSATIONAL = "CONVERSATIONAL"
    OFF_TOPIC = "OFF_TOPIC"
    BLOCKED = "BLOCKED"
    UNKNOWN = "UNKNOWN"


class JurisdictionType(str, Enum):
    INDIA = "INDIA"
    INTERNATIONAL = "INTERNATIONAL"
    BOTH = "BOTH"
    UNKNOWN = "UNKNOWN"


class FormulationType(str, Enum):
    CLASSICAL_AYURVEDIC = "CLASSICAL_AYURVEDIC"
    PROPRIETARY_AYURVEDIC = "PROPRIETARY_AYURVEDIC"
    PHYTOPHARMACEUTICAL = "PHYTOPHARMACEUTICAL"
    FOOD_AYURVEDA_AAHAR = "FOOD_AYURVEDA_AAHAR"
    COSMETIC = "COSMETIC"
    INSUFFICIENT_INFO = "INSUFFICIENT_INFO"


class GuardrailAction(str, Enum):
    PASSED = "PASSED"
    BLOCKED_OFF_TOPIC = "BLOCKED_OFF_TOPIC"
    BLOCKED_JAILBREAK = "BLOCKED_JAILBREAK"
    BLOCKED_LEGAL_ADVICE = "BLOCKED_LEGAL_ADVICE"
    BLOCKED_ILLEGAL = "BLOCKED_ILLEGAL"


class RetrievalStrategy(str, Enum):
    VECTOR = "vector"
    BM25 = "bm25"
    HYBRID = "hybrid"


class NodeStatus(str, Enum):
    STARTED = "STARTED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


# ============================================================
# SPAN INPUT/OUTPUT MODELS
# ============================================================

class PlannerInput(BaseModel):
    user_query: str = Field(..., description="Raw user query")
    conversation_length: int = Field(..., description="Number of prior messages")
    thread_id: str


class PlannerOutput(BaseModel):
    intent: IntentType
    refined_query: Optional[str] = None
    refusal_reason: Optional[str] = None
    confidence: float = Field(..., ge=0, le=1)


class JurisdictionRouterInput(BaseModel):
    query: str
    intent: IntentType


class JurisdictionRouterOutput(BaseModel):
    jurisdiction: JurisdictionType
    confidence: float = Field(..., ge=0, le=1)
    reasoning: Optional[str] = None


class FormulationClassifierInput(BaseModel):
    query: str
    jurisdiction: JurisdictionType


class FormulationClassifierOutput(BaseModel):
    formulation_type: FormulationType
    abs_required: Optional[bool] = None
    confidence: float = Field(..., ge=0, le=1)
    missing_questions: List[str] = []


class RetrievalInput(BaseModel):
    query: str
    jurisdiction: JurisdictionType
    formulation_type: Optional[FormulationType] = None
    strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    top_k: int = 15


class RetrievedChunk(BaseModel):
    content: str
    source: str
    section: Optional[str] = None
    jurisdiction: JurisdictionType
    score: float
    metadata: Dict[str, Any] = {}


class RetrievalOutput(BaseModel):
    chunks: List[RetrievedChunk]
    total_candidates: int
    reranked: bool = True
    reranker_model: str = "FlashRank"
    latency_ms: int


class LLMGenerationInput(BaseModel):
    prompt_tokens: int
    context_chunks: int
    jurisdiction: JurisdictionType
    formulation_type: Optional[FormulationType] = None
    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.1


class LLMGenerationOutput(BaseModel):
    answer: str
    completion_tokens: int
    cache_status: Literal["HIT", "MISS"] = "MISS"
    latency_ms: int
    gateway: str = "Portkey"


class CitationVerificationInput(BaseModel):
    answer: str
    retrieved_chunks: List[RetrievedChunk]


class Citation(BaseModel):
    text: str
    source: str
    section: Optional[str] = None
    verified: bool = False
    confidence: float = Field(..., ge=0, le=1)


class CitationVerificationOutput(BaseModel):
    citations: List[Citation]
    total_claims: int
    supported_claims: int
    unsupported_claims: int
    overall_confidence: float = Field(..., ge=0, le=1)


class GuardrailInput(BaseModel):
    user_query: str
    thread_id: str


class GuardrailOutput(BaseModel):
    action: GuardrailAction
    response: Optional[str] = None
    indicators_matched: List[str] = []
    latency_ms: int


class APIRequest(BaseModel):
    method: str
    path: str
    thread_id: str
    query_length: int


class APIResponse(BaseModel):
    status_code: int
    thread_id: str = "unknown"
    intent: Optional[IntentType] = None
    jurisdiction: Optional[JurisdictionType] = None
    formulation_type: Optional[FormulationType] = None
    confidence: Optional[float] = None
    abs_required: Optional[bool] = None
    citations_count: int = 0
    latency_ms: int
    error: Optional[str] = None


class IngestionRecord(BaseModel):
    source_id: str
    source_type: str
    jurisdiction: JurisdictionType
    category: str
    chunks_indexed: int
    version_date: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ============================================================
# HIGH-LEVEL EVENT MODELS (for dashboard queries)
# ============================================================

class QueryEvent(BaseModel):
    """Top-level event for each /query request — queryable in Logfire."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    thread_id: str
    user_query: str
    intent: IntentType
    jurisdiction: JurisdictionType
    formulation_type: Optional[FormulationType] = None
    guardrail_action: GuardrailAction
    retrieval_latency_ms: int
    llm_latency_ms: int
    total_latency_ms: int
    citations_count: int
    confidence: float
    abs_required: Optional[bool] = None
    status: Literal["SUCCESS", "BLOCKED", "ERROR"]
    error: Optional[str] = None


class ClassificationEvent(BaseModel):
    """Event for /classify endpoint."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    thread_id: str
    description: str
    formulation_type: FormulationType
    abs_required: Optional[bool]
    confidence: float
    latency_ms: int


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def create_query_event(**kwargs) -> QueryEvent:
    """Factory for QueryEvent with defaults."""
    return QueryEvent(**kwargs)


def create_classification_event(**kwargs) -> ClassificationEvent:
    return ClassificationEvent(**kwargs)
