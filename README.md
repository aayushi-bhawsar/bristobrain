# 🍽️ BistroBrain

> **Democratizing Enterprise Intelligence for Independent Restaurant Owners**

BistroBrain is an AI-powered "Store Manager" that transforms raw POS (Point-of-Sale) data into actionable, plain-English decisions — delivered daily via WhatsApp or mobile. No data science degree required.

---

## 🧠 The Problem

Small and mid-sized restaurant owners are **data rich but insight poor**:

- 📦 **30% average food wastage** from poor inventory tracking
- 💸 **15% profit leakage** due to inefficient staffing and pricing
- 📉 A growing competitive gap between enterprise chains and local bistros

While global chains run proprietary AI teams to optimize margins, local cafés are being squeezed out. BistroBrain bridges this gap.

---

## ✨ What It Does

BistroBrain acts as an **AI Store Manager** that:

- 📋 Ingests POS data (CSV/JSON) from Square, Toast, Petpooja
- 🤖 Runs specialized AI agents for inventory, marketing, and pricing
- 📲 Delivers **Action Cards** via WhatsApp (e.g., *"5kg salmon expires in 48h — run a Seafood Special tomorrow"*)
- 💬 Answers natural language questions: *"Why were my margins low last Tuesday?"*
- 📈 Predicts footfall using weather and local event data

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     BistroBrain                          │
├─────────────┬───────────────────────┬───────────────────┤
│  Data Layer │    Agent Orchestration │   Interface Layer │
│  (Ingestion)│       (LangGraph)      │  (WhatsApp / API) │
├─────────────┼───────────────────────┼───────────────────┤
│  POS CSV    │   InventoryAgent       │   FastAPI REST    │
│  Weather API│   MarketingAgent       │   WhatsApp Hook   │
│  Events API │   PricingAgent         │   WebSocket Q&A   │
│  RAG Store  │   InsightAgent         │   Action Cards    │
└─────────────┴───────────────────────┴───────────────────┘
```

**Stack:**
| Layer | Technology |
|---|---|
| LLM | Claude 3.5 Sonnet / GPT-4o |
| Orchestration | LangGraph |
| Fast Inference | Groq (Llama-3) |
| Vector Store | ChromaDB |
| API | FastAPI |
| Messaging | Twilio WhatsApp |
| Data | Pandas, SQLite |

---

## 📁 Project Structure

```
bistrobrain/
├── agents/
│   ├── __init__.py
│   ├── inventory_agent.py      # Flags shortages & expirations
│   ├── marketing_agent.py      # Generates promo copy from surplus
│   ├── pricing_agent.py        # Suggests dynamic pricing
│   └── insight_agent.py        # Answers natural language queries
├── api/
│   ├── __init__.py
│   ├── main.py                 # FastAPI entrypoint
│   ├── routes/
│   │   ├── action_cards.py     # Daily action card endpoint
│   │   ├── query.py            # Natural language Q&A endpoint
│   │   └── webhook.py          # WhatsApp webhook handler
├── data/
│   ├── ingestion.py            # POS data normalizer
│   ├── rag_store.py            # ChromaDB RAG pipeline
│   └── schema.py               # Pydantic models
├── utils/
│   ├── llm_client.py           # Anthropic/OpenAI unified client
│   ├── weather.py              # Weather API integration
│   └── events.py               # Local events calendar integration
├── synthetic_data/
│   ├── generate_pos_data.py    # Generate synthetic POS datasets
│   └── sample_data.csv         # 30-day sample restaurant data
├── scripts/
│   ├── run_daily_digest.py     # Cron: generate & send Action Cards
│   └── backtest.py             # Backtest AI recommendations
├── tests/
│   ├── test_agents.py
│   ├── test_ingestion.py
│   └── test_api.py
├── .env.example
├── requirements.txt
├── docker-compose.yml
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- An Anthropic API key ([get one here](https://console.anthropic.com))
- Optional: Twilio account for WhatsApp delivery
- Optional: OpenWeatherMap API key

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/bistrobrain.git
cd bistrobrain
```

### 2. Set up virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
# Open .env and fill in your API keys
```

Required keys in `.env`:

```env
ANTHROPIC_API_KEY=your_key_here
GROQ_API_KEY=your_key_here          # For fast inference
TWILIO_ACCOUNT_SID=...              # Optional: WhatsApp delivery
TWILIO_AUTH_TOKEN=...
TWILIO_WHATSAPP_FROM=whatsapp:+14155238886
OPENWEATHER_API_KEY=...             # Optional: weather correlation
```

### 5. Generate synthetic POS data

```bash
python synthetic_data/generate_pos_data.py
# Generates synthetic_data/sample_data.csv with 30 days of data
```

### 6. Ingest data and build RAG index

```bash
python -m data.ingestion --input synthetic_data/sample_data.csv
# Parses, normalizes, and indexes data into ChromaDB
```

### 7. Run the API server

```bash
uvicorn api.main:app --reload --port 8000
```

Open **http://localhost:8000/docs** for the interactive API explorer.

### 8. Run your first daily digest

```bash
python scripts/run_daily_digest.py
# Prints (or sends via WhatsApp) today's Action Cards
```

---

## 🐳 Docker (Recommended)

```bash
docker-compose up --build
```

This starts:
- `bistrobrain-api` on port 8000
- `chromadb` vector store on port 8001

---

## 💬 Example Action Cards

BistroBrain delivers cards like these every morning:

```
🟠 URGENT — Inventory
5kg salmon expires in 48h.
→ Run a "Seafood Special" for tomorrow's lunch.
   Suggested caption: "Fresh catch, fresh savings 🐟"

🟡 PRICING
Your Tuesday dinner covers are 22% below Monday avg.
→ Consider a weekday combo deal to boost covers.

🟢 WIN
Pasta dishes drove 34% of revenue last week.
→ Feature them prominently on your weekend menu.
```

---

## 🔍 Natural Language Q&A

Ask your data anything:

```
POST /api/query
{ "question": "Why were my margins low last Tuesday?" }

→ "Last Tuesday (Mar 18), food cost ratio spiked to 38% vs your 28%
   weekly average. The primary driver was a bulk chicken order (₹4,200)
   that preceded a low-revenue lunch service (₹6,100 vs avg ₹9,800).
   Recommendation: align bulk orders to high-footfall days."
```

---

## 📊 Evaluation

| Metric | Target | Method |
|---|---|---|
| Insight Accuracy | >85% | Human-in-the-loop validation |
| Voice-to-Insight Latency | <5 seconds | Groq fast inference |
| Actionability Score | >4/5 | Qualitative user testing |
| Waste Reduction | ~30% | Backtesting against failed months |

Run the backtest suite:

```bash
python scripts/backtest.py --months 3
```

---

## 🛣️ Roadmap

- [x] Core RAG pipeline + Agent orchestration
- [x] WhatsApp Action Card delivery
- [x] Natural language Q&A
- [ ] Voice interface (Whisper STT + TTS)
- [ ] ONDC integration for automated procurement
- [ ] Multi-outlet dashboard
- [ ] Pivot templates: pharmacies, boutique retail

---

## 🤝 Contributing

Pull requests are welcome! Please open an issue first to discuss what you'd like to change.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built with ❤️ for the independent restaurant community.*
