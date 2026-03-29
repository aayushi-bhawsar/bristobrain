# Contributing to BistroBrain

Thank you for your interest in improving BistroBrain! This document covers the development workflow, coding standards, and how to add new agents or integrations.

---

## Table of Contents

1. [Development Setup](#development-setup)
2. [Project Structure](#project-structure)
3. [Running Tests](#running-tests)
4. [Adding a New Agent](#adding-a-new-agent)
5. [Adding a New API Route](#adding-a-new-api-route)
6. [Code Style](#code-style)
7. [Commit Message Convention](#commit-message-convention)
8. [Pull Request Process](#pull-request-process)

---

## Development Setup

```bash
git clone https://github.com/YOUR_USERNAME/bistrobrain.git
cd bistrobrain

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# Fill in at least ANTHROPIC_API_KEY

# Seed sample data
python synthetic_data/generate_pos_data.py
python -m data.ingestion --input synthetic_data/sample_data.csv

# Start the API
uvicorn api.main:app --reload
```

---

## Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/test_agents.py -v

# Single test
pytest tests/test_agents.py::TestInventoryAgent::test_detects_expiry_alert -v

# With coverage report
pytest --cov=. --cov-report=html
open htmlcov/index.html
```

Tests are designed to run **without any real API keys** — all LLM calls are mocked. The only external dependency in the test suite is SQLite (built into Python).

---

## Adding a New Agent

All agents live in `agents/` and follow a consistent pattern:

### 1. Create the agent file

```python
# agents/my_new_agent.py
from __future__ import annotations
import uuid
from data.schema import ActionCard, ActionPriority, ActionType
from utils.llm_client import get_llm_client

SYSTEM_PROMPT = """You are BistroBrain's [Agent Name]..."""

class MyNewAgent:
    def __init__(self) -> None:
        self._llm = get_llm_client()

    def analyse(self, data: ...) -> list[ActionCard]:
        # 1. Detect conditions requiring an action card
        # 2. Build a focused LLM prompt
        # 3. Parse JSON response into ActionCard
        # 4. Always provide a fallback card if LLM fails
        ...
```

**Key rules for agents:**
- Always return `list[ActionCard]` — never raise exceptions to callers
- Always include a `try/except` around LLM calls with a non-empty fallback card
- Use `fast=True` in `llm.complete()` for time-sensitive interactive paths
- Prompt the LLM to return **JSON only** (no markdown fences in the expected output)
- Strip markdown fences defensively: `re.sub(r"```(?:json)?\s*|\s*```", "", raw)`

### 2. Register in `agents/__init__.py`

```python
from agents.my_new_agent import MyNewAgent
__all__ = [..., "MyNewAgent"]
```

### 3. Wire into the digest

Add to `api/routes/action_cards.py` and `scripts/run_daily_digest.py`.

### 4. Write tests

Add `tests/test_agents.py::TestMyNewAgent` with:
- At least one test for the detection logic (no LLM needed)
- At least one test with a mocked LLM response
- At least one test for LLM failure fallback

---

## Adding a New API Route

```python
# api/routes/my_route.py
from fastapi import APIRouter
router = APIRouter()

@router.get("/my-endpoint")
async def my_endpoint():
    ...
```

Register in `api/main.py`:

```python
from api.routes import my_route
app.include_router(my_route.router, prefix="/api", tags=["My Feature"])
```

---

## Code Style

We use **Black** for formatting and **Ruff** for linting.

```bash
black .
ruff check . --fix
```

Style guidelines:
- Type hints on all function signatures
- `from __future__ import annotations` at the top of every module
- `structlog` for all logging — no `print()` in production code
- Pydantic models for all data structures that cross module boundaries
- Environment variables accessed via `os.getenv()` with sensible defaults

---

## Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <short summary>

[optional body]
```

Types: `feat`, `fix`, `chore`, `test`, `docs`, `refactor`, `perf`

Scopes: `agents`, `api`, `data`, `utils`, `scripts`, `docker`

Examples:
```
feat(agents): add StaffingAgent for shift optimization
fix(data): handle missing food_cost column in legacy POS exports
test(api): add webhook auth bypass test for missing owner number
docs: update README with ONDC integration roadmap
```

---

## Pull Request Process

1. **Branch** from `main`: `git checkout -b feat/my-feature`
2. **Write tests first** — PRs without tests for new behaviour will be asked to add them
3. **Ensure CI passes**: `pytest && ruff check . && black --check .`
4. **Update README** if you've added a new feature, config key, or changed the architecture
5. **Open a PR** with a clear description of *what* changed and *why*
6. **One approval** from a maintainer required before merge

---

## Roadmap Items (Good First Issues)

| Feature | Difficulty | Notes |
|---|---|---|
| Staffing agent (shift cost vs covers) | Medium | See `PricingAgent` as a template |
| Voice interface (Whisper STT) | Medium | Add `/api/voice` endpoint |
| ONDC procurement integration | Hard | Auto-raise purchase orders |
| Multi-outlet support | Hard | Tenant isolation in SQLite + ChromaDB |
| WhatsApp interactive buttons | Easy | Twilio template messages |
| Petpooja direct API integration | Medium | Replace CSV import |

---

Questions? Open an issue or start a discussion on GitHub.
