from typing import TypedDict, List, Dict, Any
import os
from langgraph.graph import StateGraph, END
from datetime import datetime, timezone
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

def get_llm():
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY", ""),
        model_name=os.getenv("LLM_MODEL_NAME", "chatgpt-120b-oss")
    )

class AgentState(TypedDict):
    user_id: int
    conversation_id: int
    messages: List[Dict[str, str]]
    safety_triggered: bool
    retrieval_used: bool
    retrieved_contexts: List[str]
    memory_actions: List[str]
    notification_actions: List[str]
    reflection_summary: str
    final_reply: str

def append_message_node(state: AgentState):
    # Already appended by the API handler, just an explicit node
    return state

def safety_check_node(state: AgentState):
    last_msg = state["messages"][-1]["content"].lower()
    keywords = ["suicide", "self-harm", "panic", "hopelessness", "severe depression"]
    triggered = any(k in last_msg for k in keywords)
    return {"safety_triggered": triggered}

def retrieval_decision_node(state: AgentState):
    # Dummy logic: always retrieve if not safety triggered
    return {"retrieval_used": not state["safety_triggered"]}

def vector_search_node(state: AgentState):
    from storage import search_vector
    if not state["retrieval_used"]:
        return {"retrieved_contexts": []}
    query = state["messages"][-1]["content"]
    res = search_vector(query, user_id=state["user_id"], top_k=2)
    contexts = [r["content"] for r in res]
    return {"retrieved_contexts": contexts}

def response_generation_node(state: AgentState):
    if state["safety_triggered"]:
        reply = "I'm so sorry you're feeling this way. Please know you're not alone. I highly encourage you to reach out to a professional or a crisis hotline immediately."
    else:
        ctx_str = " ".join(state["retrieved_contexts"])
        user_msg = state["messages"][-1]["content"]
        
        system_prompt = "You are a supportive, empathetic AI best buddy. "
        if ctx_str:
            system_prompt += f"Here is relevant context about the user: {ctx_str}"
            
        try:
            llm = get_llm()
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_msg)
            ]
            response = llm.invoke(messages)
            reply = str(response.content)
        except Exception as e:
            reply = f"I hear you. (Context used: {bool(ctx_str)}). I'm here to support you."
    return {"final_reply": reply}

def memory_write_node(state: AgentState):
    # Dummy logic to store memory if they say 'remember'
    last_msg = state["messages"][-1]["content"].lower()
    actions = []
    if "remember" in last_msg:
        from storage import SessionLocal, MemoryItem
        db = SessionLocal()
        item = MemoryItem(user_id=state["user_id"], content=last_msg, category="auto", tags="[]", confidence=100)
        db.add(item)
        db.commit()
        db.close()
        actions.append("Stored memory")
    return {"memory_actions": actions}

def notification_decision_node(state: AgentState):
    # Dummy logic for notifications
    last_msg = state["messages"][-1]["content"].lower()
    actions = []
    if "terrible night" in last_msg:
        from storage import SessionLocal, NotificationTask
        db = SessionLocal()
        task = NotificationTask(
            user_id=state["user_id"],
            title="Morning check-in",
            body="How are you feeling this morning?",
            scheduled_for=datetime.now(timezone.utc),
            status="pending"
        )
        db.add(task)
        db.commit()
        db.close()
        actions.append("Scheduled morning check-in")
    return {"notification_actions": actions}

def reflection_node(state: AgentState):
    summary = "Safety handled." if state["safety_triggered"] else "Normal response."
    if state["retrieval_used"]:
        summary += " Retrieval provided context."
    return {"reflection_summary": summary}

def route_after_safety(state: AgentState):
    if state["safety_triggered"]:
        return "response_generation_node"
    return "retrieval_decision_node"

def route_after_retrieval(state: AgentState):
    if state["retrieval_used"]:
        return "vector_search_node"
    return "response_generation_node"

workflow = StateGraph(AgentState)
workflow.add_node("append_message_node", append_message_node)
workflow.add_node("safety_check_node", safety_check_node)
workflow.add_node("retrieval_decision_node", retrieval_decision_node)
workflow.add_node("vector_search_node", vector_search_node)
workflow.add_node("response_generation_node", response_generation_node)
workflow.add_node("memory_write_node", memory_write_node)
workflow.add_node("notification_decision_node", notification_decision_node)
workflow.add_node("reflection_node", reflection_node)

workflow.set_entry_point("append_message_node")
workflow.add_edge("append_message_node", "safety_check_node")
workflow.add_conditional_edges("safety_check_node", route_after_safety)
workflow.add_conditional_edges("retrieval_decision_node", route_after_retrieval)
workflow.add_edge("vector_search_node", "response_generation_node")
workflow.add_edge("response_generation_node", "memory_write_node")
workflow.add_edge("memory_write_node", "notification_decision_node")
workflow.add_edge("notification_decision_node", "reflection_node")
workflow.add_edge("reflection_node", END)

agent_app = workflow.compile()
