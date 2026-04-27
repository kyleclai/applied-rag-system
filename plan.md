# RAG-Enhanced Music Recommender — Final Project Plan

**Base project:** Module 3 — VibeFinder 1.0 (Music Recommender Simulation)  
**Extension:** Add RAG pipeline + agentic workflow + test harness + fine-tuning via few-shot prompting  
**Goal:** Simple, effective, portfolio-quality MVP covering all required features + all 4 stretch goals

---

## What's Being Extended

The original system (`src/recommender.py` + `data/songs.csv`) scores 20 songs against a user taste profile and returns a ranked list. It has no LLM, no knowledge retrieval, and no explanation quality beyond score breakdowns.

**What's being added:**
1. A music knowledge base (`music_kb/*.md`) — genre guides, mood profiles, energy descriptions
2. A RAG retriever that fetches relevant KB chunks before generating explanations
3. A Gemini LLM client that grounds answers in retrieved chunks (not free-form generation)
4. A 5-step agentic workflow with observable logged intermediate steps
5. Confidence scoring and guardrails
6. A Streamlit UI with baseline vs. RAG-enhanced mode comparison
7. A test harness that runs predefined profiles and prints pass/fail

---

## System Architecture

```
User Profile (genre, mood, energy)
         ↓
┌──────────────────────────────────────────────┐
│            RecommendationAgent               │
│  Step 1: Profile Analyzer                    │
│    → extract search terms from profile       │
│  Step 2: Knowledge Retriever                 │ ← music_kb/*.md
│    → TF-IDF inverted index retrieval         │
│  Step 3: Song Scorer (EXISTING logic)        │ ← data/songs.csv
│    → score_song() + recommend_songs()        │
│  Step 4: LLM Explainer (few-shot Gemini)     │ ← GEMINI_API_KEY
│    → explanation grounded in retrieved KB    │
│  Step 5: Confidence Assessor                 │
│    → 0.0-1.0 score + guardrail check         │
└──────────────────────────────────────────────┘
         ↓
Recommendations + Explanations + Confidence Score
         ↓
   Streamlit UI (both modes) / Test Harness
```

---

## Implementation Checklist

### Step 0: Repo setup
- [ ] Update `requirements.txt`: add `google-generativeai>=0.5.0`, `python-dotenv`
- [ ] Create `.env.example` with `GEMINI_API_KEY=your_key_here`
- [ ] Create folder structure: `music_kb/`, `evaluation/`, `logs/`, `assets/`

### Step 1: Music knowledge base (`music_kb/`)
Four plain `.md` files — no vector DB, same approach as DocuBot's `docs/` folder:
- `genres.md` — 1-2 paragraph descriptions of all genres in songs.csv (pop, rock, lofi, jazz, synthwave, hip-hop, ambient, indie pop, edm, country, classical, r&b, folk, metal, blues, reggae, electronic)
- `moods.md` — descriptions of each mood tag (happy, chill, intense, relaxed, moody, focused, confident, energetic, nostalgic, melancholic, romantic, peaceful, aggressive, soulful, uplifting, dreamy)
- `energy_profiles.md` — what low (0.0-0.4), medium (0.4-0.7), and high (0.7-1.0) energy feel like; listening contexts
- `artist_context.md` — brief notes on each artist in songs.csv

### Step 2: RAG retriever (`src/rag_retriever.py`)
Borrow inverted index pattern from DocuBot's `docubot.py`:
```python
class MusicKnowledgeRetriever:
    def __init__(self, kb_folder="music_kb")
    def load_documents() -> List[Tuple[str, str]]   # (filename, text)
    def build_index() -> Dict[str, List[str]]        # word → [filenames]
    def score_document(query, text) -> int
    def retrieve(query, top_k=4) -> List[Tuple[str, str]]  # (source, chunk)
```
Reuse DocuBot's `STOPWORDS` set as-is.

### Step 3: LLM client (`src/llm_client.py`)
Adapted from DocuBot's `llm_client.py`:
```python
GEMINI_MODEL_NAME = "gemini-2.5-flash"

class MusicLLMClient:
    def __init__()  # reads GEMINI_API_KEY, raises RuntimeError if missing
    def generate_explanation(user_profile, top_songs, retrieved_chunks) -> str
        # few-shot prompt: 2-3 example "music journalist" style explanations baked in
        # instructs model to ONLY use retrieved chunks + song metadata
    def generate_explanation_baseline(user_profile, top_songs) -> str
        # SAME but NO few-shot examples — used for fine-tuning/specialization comparison
    def rate_confidence(explanation, retrieved_chunks) -> float
        # model self-rates 0.0-1.0 based on retrieved context quality
```

### Step 4: Agentic workflow (`src/agent.py`)
```python
class RecommendationAgent:
    def run(user_profile_dict) -> AgentResult
    # Executes all 5 steps, printing + logging each:
    #   Step 1 [Profile Analysis]    → extract search terms
    #   Step 2 [Knowledge Retrieval] → retrieve chunks from music_kb
    #   Step 3 [Song Scoring]        → call recommend_songs() from recommender.py
    #   Step 4 [LLM Generation]      → call MusicLLMClient.generate_explanation()
    #   Step 5 [Confidence Check]    → score, log guardrail pass/fail
```
Logs to `logs/agent_run.log` with timestamps. Each step output goes to console.

### Step 5: Confidence scorer (`src/confidence.py`)
```python
def score_confidence(retrieved_chunks, llm_response, top_songs) -> float
    # Signal 1: retrieval coverage (genre/mood terms found in chunks?)
    # Signal 2: LLM self-rating via rate_confidence()
    # Signal 3: response length + completeness
    # Guardrail: if score < 0.4, return a safe fallback message
```

### Step 6: Streamlit app (`app.py`)
- Sidebar: user profile inputs (genre, mood, energy slider, optional decade/popularity)
- Radio: "Baseline Mode" vs "RAG-Enhanced Mode"
- RAG mode: expandable sections per agent step, top 5 songs + LLM explanation, confidence badge, source labels
- Baseline mode: original scoring output only (for comparison)

### Step 7: Test harness (`evaluation/test_harness.py`) — Stretch +2
6 predefined profiles, checks per profile:
- Top song genre matches expected
- Confidence > 0.5
- Explanation non-empty and no "I don't know"
- All 5 agent steps complete without error

Outputs: `PASS/FAIL` per test + summary line + confidence table

### Step 8: Retrieval evaluation (`evaluation/retrieval_eval.py`) — Stretch +2 (RAG Enhancement)
- Single-source (genres.md only) vs. multi-source (all 4 kb files) retrieval
- Measures retrieval hit rate and chunk relevance score
- Prints before/after table showing improvement

---

## Files to Create / Modify

```
applied-rag-system/
├── src/
│   ├── recommender.py          ← KEEP unchanged (base Module 3 code)
│   ├── main.py                 ← KEEP unchanged
│   ├── rag_retriever.py        ← NEW (borrow DocuBot pattern)
│   ├── llm_client.py           ← NEW (borrow DocuBot LLM client)
│   ├── agent.py                ← NEW (5-step orchestrator)
│   └── confidence.py           ← NEW (confidence scoring)
├── music_kb/
│   ├── genres.md               ← NEW
│   ├── moods.md                ← NEW
│   ├── energy_profiles.md      ← NEW
│   └── artist_context.md       ← NEW
├── data/
│   └── songs.csv               ← KEEP unchanged
├── evaluation/
│   ├── test_harness.py         ← NEW (test harness stretch)
│   └── retrieval_eval.py       ← NEW (RAG enhancement stretch)
├── logs/                       ← auto-created by agent.py
├── assets/                     ← NEW folder for architecture diagram
├── tests/
│   └── test_recommender.py     ← KEEP unchanged
├── app.py                      ← NEW (Streamlit UI)
├── requirements.txt            ← UPDATE
├── .env.example                ← NEW
├── README.md                   ← UPDATE (full rubric-compliant docs)
└── model_card.md               ← UPDATE (reflection prompts)
```

---

## Feature Coverage

| Requirement | Implementation |
|---|---|
| Required AI feature: RAG | Steps 2+4 — retrieve KB chunks, Gemini answers using ONLY those chunks |
| Logging + guardrails | `logs/agent_run.log` + confidence < 0.4 fallback message |
| Runs reproducibly | `.env.example` + `requirements.txt` + clear README steps |
| System diagram | Mermaid diagram exported to `assets/` |
| Reliability testing | Test harness (Step 7) + confidence scoring (Step 5) |
| RAG Enhancement (+2) | Multi-source KB + `retrieval_eval.py` showing measurable improvement |
| Agentic Workflow (+2) | 5-step agent with logged intermediate steps visible in UI + terminal |
| Fine-Tuning/Specialization (+2) | Few-shot vs. baseline prompt in `llm_client.py` — output measurably differs |
| Test Harness (+2) | `evaluation/test_harness.py` — pass/fail + confidence table for 6 profiles |

---

## Verification Steps

1. `pytest` — existing recommender tests pass unchanged
2. `python evaluation/test_harness.py` — target 5/6 pass, confidence avg > 0.6
3. `python evaluation/retrieval_eval.py` — multi-source shows higher hit rate than single-source
4. `streamlit run app.py` — test 3 profiles (pop/happy, jazz/relaxed, metal/intense) in both modes
5. Confirm `logs/agent_run.log` shows all 5 steps per run
6. Loom: record end-to-end in RAG mode with 3 different profiles
