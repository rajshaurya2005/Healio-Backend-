import json
from datetime import datetime
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from storage import (
    get_db, ChatRequest, ChatResponse, MemoryRequest, 
    VectorIngestRequest, VectorSearchRequest, NotificationRequest,
    Message, MemoryItem, VectorDocument, NotificationTask, SafetyEvent, AgentRun,
    insert_vector, search_vector, delete_vector
)
from agent import agent_app

app = FastAPI(title="Healio Backend")

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest, db: Session = Depends(get_db)):
    # 1. Save user message
    user_msg = Message(
        conversation_id=req.conversation_id,
        role="user",
        content=req.message,
        metadata_json=json.dumps(req.metadata) if req.metadata else None
    )
    db.add(user_msg)
    db.commit()

    # 2. Run graph
    initial_state = {
        "user_id": req.user_id,
        "conversation_id": req.conversation_id,
        "messages": [{"role": "user", "content": req.message}],
        "safety_triggered": False,
        "retrieval_used": False,
        "retrieved_contexts": [],
        "memory_actions": [],
        "notification_actions": [],
        "reflection_summary": "",
        "final_reply": ""
    }
    
    final_state = agent_app.invoke(initial_state)
    
    # 3. Log Safety Event if triggered
    if final_state["safety_triggered"]:
        sev = SafetyEvent(
            user_id=req.user_id,
            severity="High",
            trigger_text=req.message,
            handled=True
        )
        db.add(sev)
    
    # 4. Save agent run
    run = AgentRun(
        conversation_id=req.conversation_id,
        reflection_summary=final_state["reflection_summary"],
        retrieval_used=final_state["retrieval_used"],
        safety_triggered=final_state["safety_triggered"]
    )
    db.add(run)
    
    # 5. Save assistant message
    ai_msg = Message(
        conversation_id=req.conversation_id,
        role="assistant",
        content=final_state["final_reply"]
    )
    db.add(ai_msg)
    db.commit()

    return ChatResponse(
        reply=final_state["final_reply"],
        retrieved_contexts=final_state["retrieved_contexts"],
        memory_actions=final_state["memory_actions"],
        notification_actions=final_state["notification_actions"],
        safety_triggered=final_state["safety_triggered"]
    )

@app.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: int, db: Session = Depends(get_db)):
    msgs = db.query(Message).filter(Message.conversation_id == conversation_id).order_by(Message.created_at).all()
    return [{"id": m.id, "role": m.role, "content": m.content, "created_at": m.created_at} for m in msgs]

@app.post("/memory")
def create_memory(req: MemoryRequest, db: Session = Depends(get_db)):
    item = MemoryItem(
        user_id=req.user_id,
        content=req.content,
        category=req.category,
        tags=json.dumps(req.tags),
        confidence=100
    )
    db.add(item)
    db.commit()
    return {"status": "ok", "id": item.id}

@app.get("/memory/{user_id}")
def get_memory(user_id: int, db: Session = Depends(get_db)):
    items = db.query(MemoryItem).filter(MemoryItem.user_id == user_id).all()
    return [{"id": i.id, "content": i.content, "category": i.category, "tags": json.loads(i.tags)} for i in items]

@app.delete("/memory/{memory_id}")
def del_memory(memory_id: int, db: Session = Depends(get_db)):
    db.query(MemoryItem).filter(MemoryItem.id == memory_id).delete()
    db.commit()
    return {"status": "deleted"}

@app.post("/vector/ingest")
def vector_ingest(req: VectorIngestRequest, db: Session = Depends(get_db)):
    insert_vector(req.user_id, req.content, req.metadata)
    doc = VectorDocument(user_id=req.user_id, content=req.content, chunk_index=0, metadata_json=json.dumps(req.metadata))
    db.add(doc)
    db.commit()
    return {"status": "ingested"}

@app.post("/vector/search")
def vector_search(req: VectorSearchRequest):
    res = search_vector(req.query, req.user_id, req.top_k, req.filters)
    return {"results": res}

@app.delete("/vector/{document_id}")
def vector_delete(document_id: str):
    delete_vector(document_id)
    return {"status": "deleted"}

@app.post("/notifications")
def create_notification(req: NotificationRequest, db: Session = Depends(get_db)):
    task = NotificationTask(
        user_id=req.user_id,
        title=req.title,
        body=req.body,
        scheduled_for=datetime.fromisoformat(req.scheduled_for),
        status="pending"
    )
    db.add(task)
    db.commit()
    return {"status": "created", "id": task.id}

@app.get("/notifications/{user_id}")
def list_notifications(user_id: int, db: Session = Depends(get_db)):
    tasks = db.query(NotificationTask).filter(NotificationTask.user_id == user_id).all()
    return [{"id": t.id, "title": t.title, "scheduled_for": t.scheduled_for, "status": t.status} for t in tasks]

@app.delete("/notifications/{notification_id}")
def del_notification(notification_id: int, db: Session = Depends(get_db)):
    db.query(NotificationTask).filter(NotificationTask.id == notification_id).delete()
    db.commit()
    return {"status": "deleted"}
