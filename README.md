# Healio Backend

AI emotional support agent server.

## Setup

1) Install uv package manager
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2) Create and activate virtual environment
```bash
uv venv
source .venv/bin/activate
```

3) Install dependencies
```bash
uv pip install -e .
```

4) Setup Environment
```bash
cp .env.example .env
```

5) Run Server
```bash
uvicorn main:app --reload --port 8000
```

## API Overview

- `GET /` - Health check
- `POST /chat` - Main agent interaction
- `GET /conversations/{conversation_id}` - Get chat history
- `POST /memory` - Create memory
- `GET /memory/{user_id}` - List memories
- `DELETE /memory/{memory_id}` - Delete memory
- `POST /vector/ingest` - Add vector document
- `POST /vector/search` - Search vectors
- `DELETE /vector/{document_id}` - Delete vector
- `POST /notifications` - Schedule notification
- `GET /notifications/{user_id}` - List notifications
- `DELETE /notifications/{notification_id}` - Cancel notification

## Example Request

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "conversation_id": 1, "message": "I had a terrible night", "metadata": null}'
```
