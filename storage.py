import os
import json
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from pydantic import BaseModel, Field
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

os.environ["HF_HOME"] = os.getenv("HF_HOME", "./hf_cache")

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./healio.db")
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./chroma_db")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_utc_now():
    return datetime.now(timezone.utc)

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, index=True)
    created_at = Column(DateTime, default=get_utc_now)
    updated_at = Column(DateTime, default=get_utc_now, onupdate=get_utc_now)

class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    title = Column(String)
    created_at = Column(DateTime, default=get_utc_now)
    updated_at = Column(DateTime, default=get_utc_now, onupdate=get_utc_now)

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, index=True)
    role = Column(String)
    content = Column(Text)
    metadata_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=get_utc_now)
    updated_at = Column(DateTime, default=get_utc_now, onupdate=get_utc_now)

class MemoryItem(Base):
    __tablename__ = "memory_items"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    content = Column(Text)
    category = Column(String)
    tags = Column(String) # JSON list
    confidence = Column(Integer)
    created_at = Column(DateTime, default=get_utc_now)
    updated_at = Column(DateTime, default=get_utc_now, onupdate=get_utc_now)

class VectorDocument(Base):
    __tablename__ = "vector_documents"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    content = Column(Text)
    chunk_index = Column(Integer)
    metadata_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=get_utc_now)
    updated_at = Column(DateTime, default=get_utc_now, onupdate=get_utc_now)

class NotificationTask(Base):
    __tablename__ = "notification_tasks"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    title = Column(String)
    body = Column(Text)
    scheduled_for = Column(DateTime)
    status = Column(String)
    related_message_id = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=get_utc_now)
    updated_at = Column(DateTime, default=get_utc_now, onupdate=get_utc_now)

class SafetyEvent(Base):
    __tablename__ = "safety_events"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    severity = Column(String)
    trigger_text = Column(Text)
    handled = Column(Boolean, default=False)
    created_at = Column(DateTime, default=get_utc_now)
    updated_at = Column(DateTime, default=get_utc_now, onupdate=get_utc_now)

class AgentRun(Base):
    __tablename__ = "agent_runs"
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, index=True)
    reflection_summary = Column(Text)
    retrieval_used = Column(Boolean)
    safety_triggered = Column(Boolean)
    created_at = Column(DateTime, default=get_utc_now)
    updated_at = Column(DateTime, default=get_utc_now, onupdate=get_utc_now)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ChromaDB
chroma_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
collection = chroma_client.get_or_create_collection(name="support_vectors", embedding_function=embedding_func)

def chunk_text(text: str, max_words: int = 100) -> List[str]:
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

def insert_vector(user_id: int, content: str, meta: Dict[str, Any]):
    chunks = chunk_text(content)
    ids = []
    documents = []
    metadatas = []
    for i, chunk in enumerate(chunks):
        chunk_id = f"user_{user_id}_{datetime.now().timestamp()}_{i}"
        ids.append(chunk_id)
        documents.append(chunk)
        m = meta.copy()
        m.update({"user_id": user_id, "chunk_index": i})
        metadatas.append(m)
    collection.add(documents=documents, metadatas=metadatas, ids=ids)

def delete_vector(doc_id: str):
    collection.delete(ids=[doc_id])

def update_vector(doc_id: str, content: str, meta: Dict[str, Any]):
    collection.update(ids=[doc_id], documents=[content], metadatas=[meta])

def search_vector(query: str, user_id: Optional[int] = None, top_k: int = 3, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    where = filters or {}
    if user_id is not None:
        where["user_id"] = user_id
    if not where:
        results = collection.query(query_texts=[query], n_results=top_k)
    else:
        results = collection.query(query_texts=[query], n_results=top_k, where=where)
    out = []
    for i in range(len(results["documents"][0])):
        out.append({
            "id": results["ids"][0][i],
            "content": results["documents"][0][i],
            "metadata": results["metadatas"][0][i]
        })
    return out

# Pydantic Schemas
class ChatRequest(BaseModel):
    user_id: int
    conversation_id: int
    message: str
    metadata: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    reply: str
    retrieved_contexts: List[str]
    memory_actions: List[str]
    notification_actions: List[str]
    safety_triggered: bool

class MemoryRequest(BaseModel):
    user_id: int
    content: str
    category: str
    tags: List[str]

class VectorIngestRequest(BaseModel):
    user_id: int
    content: str
    metadata: Dict[str, Any]

class VectorSearchRequest(BaseModel):
    query: str
    user_id: Optional[int] = None
    top_k: int = 3
    filters: Optional[Dict[str, Any]] = None

class NotificationRequest(BaseModel):
    user_id: int
    title: str
    body: str
    scheduled_for: str
