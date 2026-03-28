# Text2SQL PRISM — Enterprise Text-to-SQL Agent

**PRISM Swarm with Deep Think** powered by **Google ADK (Agent Development Kit)**

A production-ready, multi-agent Text-to-SQL system that converts natural language questions into accurate, optimized SQL queries using a specialized agent swarm and chain-of-thought reasoning.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        PRISM ORCHESTRATOR                               │
│                    (Google ADK Root Agent)                              │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  Phase P: Pre-processing  (ParallelAgent)                        │  │
│  │   ├── Schema Discovery Agent    → DB schema + relationships      │  │
│  │   └── Metadata Enrichment Agent → Business context + glossary   │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                              ↓                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  Phase R: Reasoning  (SequentialAgent — Deep Think)              │  │
│  │   ├── Deep Think Query Analyzer → Chain-of-thought decomposition │  │
│  │   └── Schema Linker Agent       → Entity-to-schema mapping       │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                              ↓                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  Phase I: Intent → SQL  (Agent)                                  │  │
│  │   └── SQL Generator Agent       → Dialect-aware SQL + few-shot  │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                              ↓                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  Phase S: Synthesis  (LoopAgent, max 3 iterations)               │  │
│  │   ├── SQL Validator Agent        → Syntax/schema/security/perf   │  │
│  │   └── Query Optimizer Agent      → Performance optimization      │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                              ↓                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  Phase M: Monitoring  (Agent)                                    │  │
│  │   └── Response Formatter Agent  → Execute + NL answer           │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### PRISM = P→R→I→S→M

| Phase | Name | Agent(s) | ADK Type | Purpose |
|-------|------|----------|----------|---------|
| **P** | Pre-processing | Schema Discovery + Metadata Enrichment | `ParallelAgent` | Discover schema and business context simultaneously |
| **R** | Reasoning | Deep Think Analyzer + Schema Linker | `SequentialAgent` | Chain-of-thought query decomposition and entity mapping |
| **I** | Intent Mapping | SQL Generator | `Agent` | Generate accurate, dialect-aware SQL |
| **S** | SQL Synthesis | SQL Validator + Query Optimizer | `LoopAgent` | Iterative validation and optimization (up to 3 passes) |
| **M** | Monitoring | Response Formatter | `Agent` | Execute SQL and format results |

---

## Deep Think Approach

The **Deep Think Query Analyzer** (Phase R) uses mandatory chain-of-thought reasoning:

```
Step 1: Query Decomposition  → Break into atomic components (SELECT/filter/group/sort)
Step 2: Entity Extraction    → Identify all business entities (customers, revenue, dates)
Step 3: Ambiguity Resolution → Assess mapping confidence, document assumptions
Step 4: Complexity Assessment → SIMPLE | MODERATE | COMPLEX | ADVANCED
Step 5: Mental Execution Plan → Sketch tables, JOINs, conditions before generating
Step 6: Confidence Scoring   → 0.0-1.0 confidence with justification
```

The LoopAgent in Phase S enables **iterative self-correction** — if validation fails, the error is fed back to the generator for a corrected attempt (up to 3 iterations).

---

## Project Structure

```
text2sql/
├── config/
│   ├── settings.py          # Pydantic-based configuration
│   └── prompts.py           # Centralized prompt library
├── core/
│   ├── database.py          # Async SQLAlchemy database manager
│   ├── schema_manager.py    # Schema discovery and caching
│   └── query_context.py     # Pipeline state Pydantic models
├── agents/
│   ├── swarm/
│   │   ├── schema_agent.py      # Phase P agents
│   │   ├── query_analyzer.py    # Phase R agents (Deep Think)
│   │   ├── sql_generator.py     # Phase I agent
│   │   ├── sql_validator.py     # Phase S validator
│   │   ├── query_optimizer.py   # Phase S optimizer
│   │   └── response_formatter.py # Phase M agent
│   ├── tools/
│   │   ├── database_tools.py    # DB execution tools
│   │   ├── schema_tools.py      # Schema discovery tools
│   │   ├── validation_tools.py  # SQL validation tools
│   │   └── few_shot_tools.py    # Vector store for examples
│   ├── orchestrator.py      # PRISM swarm assembly
│   └── runner.py            # ADK Runner bridge
├── api/
│   ├── app.py               # FastAPI application
│   └── models.py            # API Pydantic models
├── data/
│   ├── examples/            # Few-shot example JSON files
│   └── schemas/             # Business glossary definitions
├── tests/
│   ├── test_validation_tools.py
│   ├── test_schema_manager.py
│   └── test_api.py
└── main.py                  # CLI entry point (Typer)
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and set:
#   GOOGLE_API_KEY=your-google-api-key
#   DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/mydb
```

### 3. Set Up Demo Database (Optional)

```bash
python main.py setup-demo
# Update DATABASE_URL in .env to point to the demo SQLite database
```

### 4. Load Few-Shot Examples

```bash
python main.py load-examples ./data/examples/sample_examples.json
```

### 5. Start the API Server

```bash
python main.py serve
# API available at http://localhost:8080
# Docs at http://localhost:8080/docs
```

---

## Usage

### CLI

```bash
# Run a query
python main.py query "What are the top 10 customers by revenue this month?"

# View database schema
python main.py schema

# Start API server
python main.py serve --port 8080
```

### REST API

```bash
# Query endpoint
curl -X POST http://localhost:8080/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the total revenue by product category?",
    "database_name": "default",
    "max_rows": 100,
    "execute_query": true
  }'
```

**Response:**
```json
{
  "success": true,
  "session_id": "abc123",
  "query": "What is the total revenue by product category?",
  "generated_sql": "SELECT p.category, SUM(oi.quantity * oi.unit_price * (1 - oi.discount)) AS total_revenue\nFROM products p\nJOIN order_items oi ON p.product_id = oi.product_id\nGROUP BY p.category\nORDER BY total_revenue DESC",
  "columns": ["category", "total_revenue"],
  "rows": [
    {"category": "Services", "total_revenue": 8399.72},
    {"category": "Electronics", "total_revenue": 3499.20},
    {"category": "Software", "total_revenue": 999.80}
  ],
  "row_count": 3,
  "answer": "Services generate the highest revenue at $8,399.72, followed by Electronics at $3,499.20.",
  "confidence": 0.92,
  "pipeline_time_ms": 3241
}
```

### Python SDK

```python
import asyncio
from agents.runner import run_prism_query

async def main():
    result = await run_prism_query(
        query="Which customers haven't ordered in the last 60 days?",
        database_name="production",
        max_rows=50,
    )
    print(result["generated_sql"])
    print(result["answer"])

asyncio.run(main())
```

---

## Configuration

Key settings in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_API_KEY` | required | Google AI API key |
| `DATABASE_URL` | SQLite | Database connection URL |
| `QUERY_ANALYZER_MODEL` | `gemini-2.5-pro` | Deep Think model (Pro recommended) |
| `SQL_GENERATOR_MODEL` | `gemini-2.5-pro` | SQL generation model |
| `DEEP_THINK_MAX_ITERATIONS` | `3` | Max validation/refinement loops |
| `DEEP_THINK_CONFIDENCE_THRESHOLD` | `0.85` | Min confidence before flagging |
| `ENABLE_FEW_SHOT_EXAMPLES` | `true` | Use vector store for examples |
| `MAX_ROWS_RETURN` | `1000` | Safety limit on result rows |

---

## Supported Databases

| Database | Driver | Status |
|----------|--------|--------|
| PostgreSQL | `asyncpg` | Full support |
| MySQL | `aiomysql` | Full support |
| SQLite | `aiosqlite` | Full support (dev/demo) |
| BigQuery | `sqlalchemy-bigquery` | Supported |
| Snowflake | `snowflake-sqlalchemy` | Supported |

---

## Security

The PRISM system enforces multiple security layers:

- **SQL Allowlist**: Only `SELECT`, `WITH`, and `EXPLAIN` statements allowed
- **DDL/DML Blocking**: DROP, CREATE, INSERT, UPDATE, DELETE are rejected
- **Injection Detection**: Comment injection and UNION abuse patterns detected
- **System Table Protection**: Access to `pg_shadow`, `mysql.user`, etc. blocked
- **Row Limits**: All queries limited to `MAX_ROWS_RETURN` rows
- **Query Timeout**: Configurable statement-level timeouts per dialect

---

## Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

---

## Agent Framework

Built on **Google ADK** (`google-adk`):

- `Agent` — Individual specialized agents with tools and instructions
- `ParallelAgent` — Phase P: schema + metadata discovery in parallel
- `SequentialAgent` — Phase R: Deep Think chain (analyzer → linker)
- `LoopAgent` — Phase S: Iterative validation + optimization (max 3 passes)
- `Runner` — Async agent execution with session management
- `InMemorySessionService` — Conversation state management

The Deep Think agents use `gemini-2.5-pro` for maximum reasoning depth, while faster agents use `gemini-2.0-flash` for efficiency.
