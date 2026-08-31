from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from app.agents.state import AgentState
from app.agents.nodes.planner import planner_node
from app.agents.nodes.retriever import retrieve_node
from app.agents.nodes.responder import generate_node
from app.agents.nodes.jurisdiction_router import jurisdiction_router_node
from app.agents.nodes.formulation_classifier import formulation_classifier_node
from app.agents.nodes.citation_verifier import citation_verifier_node


workflow = StateGraph(AgentState)

workflow.add_node("planner", planner_node)
workflow.add_node("jurisdiction_router", jurisdiction_router_node)
workflow.add_node("formulation_classifier", formulation_classifier_node)
workflow.add_node("retriever", retrieve_node)
workflow.add_node("responder", generate_node)
workflow.add_node("citation_verifier", citation_verifier_node)


def route_planner(state: AgentState):
    intent = state.get("intent", "")
    if intent == "OFF_TOPIC":
        return "end"
    if intent == "CONVERSATIONAL":
        return "responder"
    if intent == "FORMULATION_CLASSIFICATION":
        return "formulation_classifier"
    return "jurisdiction_router"


def route_after_classifier(state: AgentState):
    return "jurisdiction_router"


def route_after_jurisdiction(state: AgentState):
    return "retriever"


def route_after_retriever(state: AgentState):
    return "responder"


def route_after_responder(state: AgentState):
    intent = state.get("intent", "")
    if intent == "CONVERSATIONAL":
        return "end"
    return "citation_verifier"


workflow.set_entry_point("planner")

workflow.add_conditional_edges(
    "planner",
    route_planner,
    {
        "end": END,
        "responder": "responder",
        "formulation_classifier": "formulation_classifier",
        "jurisdiction_router": "jurisdiction_router"
    }
)

workflow.add_conditional_edges(
    "formulation_classifier",
    route_after_classifier,
    {"jurisdiction_router": "jurisdiction_router"}
)

workflow.add_conditional_edges(
    "jurisdiction_router",
    route_after_jurisdiction,
    {"retriever": "retriever"}
)

workflow.add_conditional_edges(
    "retriever",
    route_after_retriever,
    {"responder": "responder"}
)

workflow.add_conditional_edges(
    "responder",
    route_after_responder,
    {"end": END, "citation_verifier": "citation_verifier"}
)

workflow.add_edge("citation_verifier", END)

checkpointer = MemorySaver()
rag_agent = workflow.compile(checkpointer=checkpointer)
