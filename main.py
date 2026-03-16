"""
AutoPrompt — FastAPI + LangGraph + Claude
Run: uvicorn main:app --reload
"""

import os
import json
import asyncio
import operator
from typing import Annotated, Optional
from dataclasses import dataclass

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from anthropic import Anthropic
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

# ── Claude client ────────────────────────────────────────────────────────────

def get_client(api_key: str) -> Anthropic:
    return Anthropic(api_key=api_key)

def ask(client: Anthropic, system: str, user: str, temperature: float = 0.7) -> str:
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return msg.content[0].text.strip()

# ── State ────────────────────────────────────────────────────────────────────

class EvalCase(TypedDict):
    user_message: str
    expected: str

class Experiment(TypedDict):
    id: int
    mutation: str
    prompt: str
    score: float
    status: str  # baseline | kept | discarded

class State(TypedDict):
    api_key: str
    base_prompt: str
    eval_cases: list[EvalCase]
    num_mutations: int

    best_prompt: str
    best_score: float
    baseline_score: float
    experiments: Annotated[list[Experiment], operator.add]

    iteration: int
    reflection: str
    candidate_prompt: str
    candidate_mutation: str
    candidate_score: float
    done: bool

# ── Scoring ──────────────────────────────────────────────────────────────────

def score_prompt(client: Anthropic, prompt: str, eval_cases: list[EvalCase]) -> float:
    scores = []
    for case in eval_cases:
        try:
            response = ask(client, prompt, case["user_message"], temperature=0.1)
            judge = ask(
                client,
                "You are an evaluator. Score how well a response meets the expected criteria. "
                "Return ONLY a float 0.0-1.0. Nothing else.",
                f'User: "{case["user_message"]}"\nExpected: "{case["expected"]}"\nResponse: "{response}"\nScore:',
                temperature=0.0,
            )
            scores.append(max(0.0, min(1.0, float(judge.split()[0]))))
        except Exception:
            scores.append(0.5)
    return round(sum(scores) / len(scores), 4) if scores else 0.5

# ── Nodes ────────────────────────────────────────────────────────────────────

def node_baseline(state: State) -> dict:
    client = get_client(state["api_key"])
    score = score_prompt(client, state["base_prompt"], state["eval_cases"])
    exp: Experiment = {"id": 0, "mutation": "baseline", "prompt": state["base_prompt"], "score": score, "status": "baseline"}
    return {
        "best_prompt": state["base_prompt"],
        "best_score": score,
        "baseline_score": score,
        "experiments": [exp],
        "iteration": 1,
        "reflection": "",
        "candidate_prompt": state["base_prompt"],
        "candidate_mutation": "",
        "candidate_score": score,
        "done": False,
    }

def node_reflect(state: State) -> dict:
    client = get_client(state["api_key"])
    history = "\n".join(
        f"  [{e['status'].upper():9s}] {e['score']*100:.0f}%  {e['mutation']}"
        for e in state["experiments"][-5:]
    ) or "  (none yet)"
    reflection = ask(
        client,
        "You are an expert prompt engineer. Analyze these prompt experiments and suggest ONE concrete direction to improve next. Be brief (2-3 sentences).",
        f"Best prompt:\n\"\"\"{state['best_prompt']}\"\"\"\n\nEval cases:\n{json.dumps(state['eval_cases'], indent=2)}\n\nExperiments (best={state['best_score']*100:.1f}%):\n{history}\n\nWhat should we try next?",
        temperature=0.5,
    )
    return {"reflection": reflection}

def node_mutate(state: State) -> dict:
    client = get_client(state["api_key"])
    try:
        raw = ask(
            client,
            'You are a prompt engineer. Generate ONE improved system prompt mutation.\nRespond ONLY with JSON: {"mutation": "what changed", "prompt": "full new prompt"}',
            f"Current best:\n\"\"\"{state['best_prompt']}\"\"\"\n\nDirection:\n{state['reflection']}",
            temperature=0.9,
        )
        raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        data = json.loads(raw)
        return {"candidate_mutation": data["mutation"], "candidate_prompt": data["prompt"]}
    except Exception:
        return {"candidate_mutation": "failed", "candidate_prompt": state["best_prompt"]}

def node_score(state: State) -> dict:
    client = get_client(state["api_key"])
    score = score_prompt(client, state["candidate_prompt"], state["eval_cases"])
    return {"candidate_score": score}

def node_judge(state: State) -> dict:
    i = state["iteration"]
    score = state["candidate_score"]
    best = state["best_score"]

    if score > best:
        status, new_best_prompt, new_best_score = "kept", state["candidate_prompt"], score
    else:
        status, new_best_prompt, new_best_score = "discarded", state["best_prompt"], best

    exp: Experiment = {
        "id": i,
        "mutation": state["candidate_mutation"],
        "prompt": state["candidate_prompt"],
        "score": score,
        "status": status,
    }
    next_i = i + 1
    return {
        "experiments": [exp],
        "best_prompt": new_best_prompt,
        "best_score": new_best_score,
        "iteration": next_i,
        "done": next_i > state["num_mutations"],
    }

def route(state: State) -> str:
    return "end" if state["done"] else "reflect"

# ── Graph ────────────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(State)
    g.add_node("baseline", node_baseline)
    g.add_node("reflect",  node_reflect)
    g.add_node("mutate",   node_mutate)
    g.add_node("score",    node_score)
    g.add_node("judge",    node_judge)
    g.add_edge(START,      "baseline")
    g.add_edge("baseline", "reflect")
    g.add_edge("reflect",  "mutate")
    g.add_edge("mutate",   "score")
    g.add_edge("score",    "judge")
    g.add_conditional_edges("judge", route, {"reflect": "reflect", "end": END})
    return g.compile()

GRAPH = build_graph()

# ── FastAPI ──────────────────────────────────────────────────────────────────

app = FastAPI(title="AutoPrompt")

class RunRequest(BaseModel):
    api_key: str
    base_prompt: str
    eval_cases: list[EvalCase]
    num_mutations: int = 5

@app.post("/run/stream")
async def run_stream(req: RunRequest):
    """Stream SSE events as experiments complete."""

    initial: State = {
        "api_key":          req.api_key,
        "base_prompt":      req.base_prompt,
        "eval_cases":       req.eval_cases,
        "num_mutations":    req.num_mutations,
        "best_prompt":      req.base_prompt,
        "best_score":       0.0,
        "baseline_score":   0.0,
        "experiments":      [],
        "iteration":        0,
        "reflection":       "",
        "candidate_prompt": req.base_prompt,
        "candidate_mutation": "",
        "candidate_score":  0.0,
        "done":             False,
    }

    async def event_stream():
        loop = asyncio.get_event_loop()
        try:
            async for chunk in GRAPH.astream(initial, stream_mode="updates"):
                for node_name, update in chunk.items():
                    # Send node start event
                    yield f"data: {json.dumps({'type': 'node', 'node': node_name, 'data': _safe(update)})}\n\n"
                    await asyncio.sleep(0)
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

def _safe(obj):
    """Make state update JSON-serializable."""
    if isinstance(obj, dict):
        return {k: _safe(v) for k, v in obj.items() if k != "api_key"}
    if isinstance(obj, list):
        return [_safe(i) for i in obj]
    return obj

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("index.html") as f:
        return f.read()