from typing import TypedDict, List, Annotated, Optional
import operator


class AgentState(TypedDict):
    # Using Annotated with operator.add ensures that messages 
    # are appended to the history rather than replaced.
    messages: Annotated[List[dict], operator.add]
    current_query: str
    documents: List[str]
    plan: List[str]
    status: str
    final_answer: str
    intent: Optional[str]
    jurisdiction: Optional[str]
    requested_jurisdiction: Optional[str]
    formulation_type: Optional[str]
    citations: List[dict]
    confidence_score: Optional[float]
    abs_required: Optional[bool]
    abs_helper: Optional[dict]
    tkdl_prior_art_pointer: Optional[dict]
    jurisdiction_answer_sets: Optional[dict]
    escalation: Optional[dict]
    refusal_reason: Optional[str]
