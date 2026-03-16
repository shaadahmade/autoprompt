# AutoPrompt

Autonomous prompt optimizer. Give it a system prompt and eval cases — it mutates, scores, and keeps improvements overnight while you sleep.

Built with **FastAPI** · **LangGraph** · **Claude** (claude-sonnet-4-6)

![graph](https://img.shields.io/badge/graph-baseline→reflect→mutate→score→judge-00e5a0?style=flat-square)
![python](https://img.shields.io/badge/python-3.10+-blue?style=flat-square)
![license](https://img.shields.io/badge/license-MIT-gray?style=flat-square)

---

## How it works

The same loop as [autoresearch](https://github.com/karpathy/autoresearch) — but for prompts instead of model weights.

```
START
  │
  ▼
baseline ── score your original prompt against all eval cases
  │
  ▼
reflect ─── Claude analyzes what's working and picks a direction
  │
  ▼
mutate ──── Claude generates one improved mutation
  │
  ▼
score ────── run the candidate against all eval cases, get a score
  │
  ▼
judge ────── if score > best → keep, else discard
  │
  ├── loop back to reflect (next mutation)
  │
  └── END (after N mutations)
```

Each node streams results to the browser via SSE as it completes. You watch experiments appear live.

---

## Quick start

**1. Clone and install**

```bash
git clone https://github.com/yourname/autoprompt
cd autoprompt
pip install -r Requirement.txt
```

**2. Run**

```bash
uvicorn main:app --reload
```

**3. Open** `http://localhost:8000`

Enter your Anthropic API key in the UI — no `.env` file needed.

---

## Project structure

```
autoprompt/
├── main.py          # FastAPI backend + LangGraph graph
├── index.html       # UI (served by FastAPI at /)
├── requirements.txt
└── README.md
```

`main.py` is the whole backend — ~200 lines. No separate files for graph, nodes, or config.

---

## Usage

1. **Paste your API key** — Anthropic key (sk-ant-...)
2. **Write your base prompt** — the system prompt you want to improve
3. **Add eval cases** — pairs of user messages + what a good response looks like
4. **Set mutations** — how many experiments to run (5–20 is a good range)
5. **Click Run** — watch experiments stream in live

When done:
- The best prompt is shown in the UI and ready to copy
- Every experiment is logged with its score and status

---

## The graph nodes

| Node | Model | What it does |
|------|-------|--------------|
| `baseline` | claude-sonnet-4-6 | Scores your original prompt as the reference point |
| `reflect` | claude-sonnet-4-6 | Analyzes experiment history and picks a direction |
| `mutate` | claude-sonnet-4-6 | Generates one concrete prompt mutation as JSON |
| `score` | claude-sonnet-4-6 | Runs candidate against all eval cases, LLM-judges each |
| `judge` | — | Keeps if score > best, discards otherwise, loops or ends |

---

## Eval cases

The quality of your eval cases determines the quality of your results. Bad evals → fake 95%+ baseline → optimizer finds nothing.

**Weak eval case:**
```json
{
  "user_message": "Help me",
  "expected": "be helpful"
}
```

**Strong eval case:**
```json
{
  "user_message": "I've been a customer for 10 years and this is the third time this has happened. I want a manager and compensation NOW.",
  "expected": "acknowledge frustration, validate loyalty, escalate to manager, offer concrete compensation without being defensive"
}
```

Hard evals with specific, measurable expected outputs give the optimizer room to find real improvements.

---

## API

One endpoint:

```
POST /run/stream
Content-Type: application/json

{
  "api_key": "sk-ant-...",
  "base_prompt": "You are a helpful assistant.",
  "eval_cases": [
    { "user_message": "...", "expected": "..." }
  ],
  "num_mutations": 5
}
```

Returns a **Server-Sent Events** stream. Each event is a JSON object:

```jsonc
// node update
{ "type": "node", "node": "reflect", "data": { "reflection": "..." } }

// experiment completed (from judge node)
{ "type": "node", "node": "judge", "data": { "experiments": [...], "best_prompt": "..." } }

// stream finished
{ "type": "done" }

// error
{ "type": "error", "message": "..." }
```

---

## Requirements

```
fastapi
uvicorn
anthropic
langgraph
langchain-anthropic
langchain-core
python-dotenv
```

Python 3.10+ required (uses `TypedDict` with `list[...]` syntax).

---

## Ideas for improvement

- **Eval library** — pre-built eval sets per industry (support, legal, sales)
- **Parallel scoring** — run eval cases concurrently to speed up scoring
- **History persistence** — save runs to SQLite, compare across sessions
- **Export** — download results as TSV or copy best prompt with one click
- **Multi-model** — use a cheaper model for scoring, smarter model for mutation

---

## License

MIT
