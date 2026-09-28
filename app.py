import os
import time
import sys
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from hindsight_client import Hindsight
from groq import Groq

# UTF-8 for console logging
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

app = FastAPI(title="Deal Intelligence Agent - Hindsight vs Stateless Web App")

# API Keys & Clients from Environment Variables
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
HINDSIGHT_API_KEY = os.environ.get("HINDSIGHT_API_KEY", "")
HINDSIGHT_BASE_URL = os.environ.get("HINDSIGHT_BASE_URL", "https://api.hindsight.vectorize.io")

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
hindsight = Hindsight(base_url=HINDSIGHT_BASE_URL, api_key=HINDSIGHT_API_KEY) if HINDSIGHT_API_KEY else None


DEFAULT_MODEL = "llama3-8b-8192"
BANK_ID = "deal_intelligence_demo"

@app.on_event("startup")
def startup_event():
    try:
        res = hindsight.create_bank(
            bank_id=BANK_ID,
            name="Deal Intelligence Assistant",
            mission="Remember client objections, past deal context, and technical hurdles."
        )
    except Exception:
        pass


def generate_groq_completion(prompt: str, target_model: str = DEFAULT_MODEL):
    candidate_models = [target_model, "openai/gpt-oss-20b", "llama-3.3-70b-versatile", "qwen/qwen3.8-27b", "openai/gpt-oss-120b"]
    for m in candidate_models:
        try:
            res = groq_client.chat.completions.create(
                model=m,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            return res.choices[0].message.content, m
        except Exception as e:
            err_str = str(e).lower()
            if "not_found" in err_str or "does not exist" in err_str or "decommissioned" in err_str:
                continue
            else:
                raise e
    raise RuntimeError("No operational Groq model found.")

class RetainRequest(BaseModel):
    content: str
    context: Optional[str] = "Client interaction note"

class CompareRequest(BaseModel):
    query: str
    recall_query: Optional[str] = None

@app.post("/api/retain")
def retain_memory(req: RetainRequest):
    if not req.content.strip():
        raise HTTPException(status_code=400, detail="Content cannot be empty")
    try:
        hindsight.retain(
            bank_id=BANK_ID,
            content=req.content.strip(),
            context=req.context or "Client interaction note"
        )
        return {"status": "success", "message": "Memory retained successfully in Hindsight."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/memories")
def search_memories(query: str = "security compliance objection Acme Corp"):
    try:
        resp = hindsight.recall(bank_id=BANK_ID, query=query)
        snippets = []
        if hasattr(resp, 'results') and resp.results:
            for r in resp.results:
                txt = getattr(r, 'text', str(r)).strip()
                if txt and txt not in snippets:
                    snippets.append(txt)
        return {"query": query, "count": len(snippets), "memories": snippets}
    except Exception as e:
        return {"query": query, "count": 0, "memories": [], "error": str(e)}

@app.post("/api/compare")
def compare_agent(req: CompareRequest):
    start_time = time.time()
    
    # 1. Stateless Prompt
    stateless_prompt = (
        "You are an enterprise sales deal assistant. "
        f"Current situation: {req.query} "
        "Provide a concise, high-impact sales pitch and meeting agenda for the sales rep."
    )
    stateless_output, model_used = generate_groq_completion(stateless_prompt)
    stateless_time = round(time.time() - start_time, 2)

    # 2. Hindsight Recall Phase
    recall_q = req.recall_query if req.recall_query else req.query
    recall_start = time.time()
    recalled_snippets = []
    try:
        recall_resp = hindsight.recall(bank_id=BANK_ID, query=recall_q)
        if hasattr(recall_resp, 'results') and recall_resp.results:
            for r in recall_resp.results:
                txt = getattr(r, 'text', str(r)).strip()
                if txt and txt not in recalled_snippets:
                    recalled_snippets.append(txt)
    except Exception as e:
        recalled_snippets = [f"Note: Error during Hindsight recall ({str(e)})"]
    
    recall_duration = round(time.time() - recall_start, 2)

    # 3. Memory-Augmented Prompt (After State)
    memories_formatted = "\n- ".join(recalled_snippets[:4]) if recalled_snippets else "No past memories found."
    stateful_prompt = f"""You are an expert deal intelligence assistant for enterprise sales teams.

HISTORICAL CONTEXT RECALLED FROM HINDSIGHT MEMORY BANK:
- {memories_formatted}

CURRENT SITUATION / QUESTION FROM SALES REP:
{req.query}

INSTRUCTIONS:
Provide a highly tailored, high-conversion sales strategy and meeting agenda. 
Directly address the client's past objections, technical requirements, and specific concerns retrieved from memory to win the deal."""

    hindsight_output, model_used_after = generate_groq_completion(stateful_prompt)
    total_time = round(time.time() - start_time, 2)

    return {
        "query": req.query,
        "stateless_response": stateless_output,
        "hindsight_response": hindsight_output,
        "recalled_memories": recalled_snippets[:4],
        "model_used": model_used,
        "timing": {
            "stateless_sec": stateless_time,
            "recall_sec": recall_duration,
            "total_sec": total_time
        }
    }

@app.get("/", response_class=HTMLResponse)
def get_dashboard():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
