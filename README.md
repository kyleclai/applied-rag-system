# 🎵 VibeFinder RAG — Applied AI System

## Base Project

This project extends **Module 3 — Music Recommender Simulation (VibeFinder 1.0)**.

The original system (`src/recommender.py`) was a content-based music recommender that scored a 20-song catalog against a user's stated taste profile using weighted signals (genre match, mood match, energy proximity). It produced ranked results with score breakdowns but had no language model, no knowledge retrieval, and no natural-language explanation capability.

## Project Summary

VibeFinder RAG extends the original recommender into a full **Retrieval-Augmented Generation (RAG)** system with an agentic multi-step pipeline. When a user provides their music taste profile, the system:

1. Extracts search terms from the profile
2. Retrieves relevant chunks from a music knowledge base (`music_kb/*.md`)
3. Scores all 20 songs using the original weighted algorithm
4. Asks Gemini 2.5 Flash to generate grounded, music-journalist-style explanations using *only* the retrieved context
5. Assesses confidence and applies a guardrail if the explanation is insufficiently grounded

The result is a system that not only ranks songs but *explains* why they fit — with explanations backed by retrieved facts, not free-form generation.

---

## Demo Walkthrough

> 🎥 **[Loom video walkthrough — add link here before submission]**

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
│    → TF-IDF inverted index retrieval         │   (genres, moods,
│                                              │    energy, artists)
│  Step 3: Song Scorer (original Module 3)     │ ← data/songs.csv
│    → score_song() + recommend_songs()        │
│  Step 4: LLM Explainer (few-shot Gemini)     │ ← GEMINI_API_KEY
│    → grounded explanation using KB chunks    │
│  Step 5: Confidence Assessor                 │
│    → 0.0–1.0 score + guardrail at 0.4        │
└──────────────────────────────────────────────┘
         ↓
Recommendations + AI Explanations + Confidence Score
         ↓
   Streamlit UI (3 modes) / CLI / Test Harness
```

The system also supports a **Fine-Tuning Comparison** mode that shows few-shot vs. baseline LLM output side by side, and a **Baseline** mode that runs only the original VibeFinder 1.0 scoring for direct comparison.

---

## Setup

### 1. Clone or copy the repo

```bash
git clone <your-repo-url>
cd applied-rag-system
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your API key

```bash
cp .env.example .env
# Edit .env and add your Gemini API key
# Get one free at: https://aistudio.google.com/app/apikey
```

### 5. Run the Streamlit app

```bash
streamlit run app.py
```

Or run CLI tools:

```bash
# Original Module 3 recommender (no LLM)
python -m src.main

# Test harness
python evaluation/test_harness.py

# Retrieval evaluation (no API key needed)
python evaluation/retrieval_eval.py
```

---

## Sample Interactions

### Example 1 — Pop / Happy / High Energy

**Profile:** `genre=pop | mood=happy | energy=0.8`

**Top 5 songs (scored):**
```
#1  Sunrise City  — Neon Echo         pop/happy     score 3.98
#2  Gym Hero      — Max Pulse         pop/intense   score 2.87
#3  Rooftop Lights — Indigo Parade    indie pop/happy score 1.96
#4  Block Party Anthem — Crowd Control hip-hop/confident 1.78
#5  Overdrive     — Pulse Grid        edm/energetic  score 1.93
```

**AI Explanation (RAG):**
> Sunrise City is practically purpose-built for your taste — Neon Echo delivers pop's signature melodic clarity with irresistible happy energy at 0.82. The production feels current without being gimmicky. Gym Hero lands in second: it shares your genre but trades some happiness for intensity, pushing to 0.93 energy — perfect if your mood shifts toward wanting to push harder. Rooftop Lights brings indie pop's textural personality to the same happy mood, feeling effortless rather than driven.

**Confidence:** 0.70 | **Guardrail:** PASSED

---

### Example 2 — Jazz / Relaxed / Low Energy

**Profile:** `genre=jazz | mood=relaxed | energy=0.37`

**Top 5 songs (scored):**
```
#1  Coffee Shop Stories — Slow Stereo  jazz/relaxed  score 4.00
#2  Adagio in Grey — Vienna Strings    classical/melancholic score 1.13
#3  River Bottom Blues — Dusty Hale   blues/soulful  score 1.09
#4  Library Rain — Paper Lanterns     lofi/chill     score 0.98
#5  Campfire Lullaby — Cedar & Ash    folk/peaceful  score 0.71
```

**AI Explanation (RAG):**
> Coffee Shop Stories achieves a rare perfect score — Slow Stereo's jazz warmth, relaxed mood, and 0.37 energy align precisely with your profile. Jazz's conversational quality, where instruments respond to each other in real time, is exactly what this track delivers. Adagio in Grey reaches slightly outside your genre into classical, but its meditative low energy and emotional depth make it a natural companion for a quiet evening.

**Confidence:** 0.70 | **Guardrail:** PASSED

---

### Example 3 — Metal / Aggressive / High Energy

**Profile:** `genre=metal | mood=aggressive | energy=0.97`

**Top 5 songs (scored):**
```
#1  Iron Curtain — Wrath Engine       metal/aggressive score 4.00
#2  Overdrive    — Pulse Grid         edm/energetic    score 1.96
#3  Storm Runner — Voltline           rock/intense     score 1.06
#4  Gym Hero     — Max Pulse          pop/intense      score 0.90
#5  Block Party Anthem — Crowd Control hip-hop/confident 0.87
```

**AI Explanation (RAG):**
> Iron Curtain is the catalog's most extreme track — Wrath Engine operates at maximum sonic intensity (0.97 energy, 174 BPM, metal/aggressive), and it scores a perfect match against your profile. Metal provides cathartic emotional release through extreme sonic density, and this track delivers exactly that. Overdrive is the only other high-energy option that comes close in raw intensity, though EDM's energetic euphoria is a different kind of force than metal's aggression.

**Confidence:** 0.70 | **Guardrail:** PASSED

---

## Design Decisions

### Why RAG over pure LLM generation?
Free-form LLM generation hallucinates artist facts and genre history. RAG constrains the model to use only what was retrieved from the knowledge base — the explanation can only cite facts that exist in `music_kb/`. This is enforced both in the prompt ("use ONLY the retrieved context") and observable via the confidence score.

### Why keep the original scoring algorithm?
The Module 3 scoring (`score_song`, `recommend_songs`) already works well for ranking. RAG adds the *explanation layer* on top — it doesn't replace the ranking logic, it enriches it. The Baseline mode lets users see the difference directly.

### Why a TF-IDF inverted index instead of vector embeddings?
Vector embeddings (e.g., Gemini's text-embedding-004) require an API call per document per query. An inverted index runs entirely offline, is instantaneous, and is sufficient for a 4-document knowledge base with rich keyword overlap. The tradeoff is semantic gap: "energetic" won't retrieve "high BPM" unless both terms appear explicitly. The knowledge base was written with this in mind (energy profile descriptions use all the relevant terms).

### Why few-shot prompting for the "fine-tuning" stretch?
Full fine-tuning requires a training dataset and model-weight updates. Few-shot prompting achieves specialized behavior with 2-3 examples in the prompt — enough to establish consistent "music journalist" tone and grounding behavior without infrastructure. The Fine-Tuning Comparison mode makes this difference visible side-by-side.

### Confidence scoring design
Three signals are blended: retrieval coverage (how many chunks were found), LLM self-rating (the model judges its own groundedness), and response completeness (non-empty, substantive length). The guardrail threshold (0.4) is conservative — it triggers only when retrieval fails AND the LLM self-rates low AND the response is short.

---

## Testing Summary

### Unit tests
```
pytest tests/     →  2/2 PASSED  (original recommender logic, unchanged)
```

### Test harness (6 profiles)
```
python evaluation/test_harness.py   →  6/6 PASSED
Average confidence: 0.70
```
All 6 predefined profiles (pop/happy, lofi/chill, rock/intense, hip-hop/confident, jazz/relaxed, metal/aggressive) returned the correct top genre match, passed the confidence threshold (≥0.50), produced non-empty explanations, and completed all 5 agent steps.

### Retrieval evaluation
```
python evaluation/retrieval_eval.py
Single-source  hit rate: 100%   avg coverage: 0.46
Multi-source   hit rate: 75%    avg coverage: 0.54
```
Multi-source retrieval shows +17% higher average coverage score, demonstrating that the richer knowledge base provides more relevant context when the right documents are retrieved. The lower hit rate in multi-source reveals a real limitation: with 4 competing documents, high-scoring artist context chunks can displace expected genre/mood files in the top-3 document window — an area for future improvement.

### What worked, what didn't, what I learned
- **Worked:** The inverted index retrieves relevant chunks reliably for genre+mood+energy queries. Confidence scoring is stable and the guardrail triggers correctly on empty retrieval.
- **Didn't work as expected:** Multi-source retrieval has a lower "hit rate" than single-source by the strict metric, which seems counterintuitive. It reveals that more documents means more competition for limited retrieval slots — a real RAG design tradeoff.
- **Learned:** RAG design requires careful attention to both the retrieval mechanism AND the knowledge base content. The KB files had to be written with retrieval in mind (making sure every genre/mood term appears explicitly) for the keyword index to work.

---

## Reflection

### What this project taught me about AI
Building VibeFinder RAG made concrete something I understood abstractly: the difference between a model *knowing* something and a system *reliably accessing* it. The original VibeFinder scored songs correctly but couldn't explain *why* in any meaningful way — it knew the rules, but couldn't articulate them. Adding RAG didn't just add explanations; it forced me to externalize knowledge into a form the system could retrieve and verify, rather than hoping the LLM would remember it correctly. That act of knowledge engineering — deciding what goes in the KB, how it's chunked, which terms need to appear — turned out to be as important as the retrieval algorithm itself.

The confidence scoring was the most surprising component to build. I expected it to be simple, but it revealed real complexity: a high-confidence score doesn't mean the explanation is *correct*, only that it's *grounded*. The system can still produce a fluent, well-grounded explanation that a music expert would find shallow. That gap between groundedness and accuracy is where real AI reliability work happens.

---

## Files

| File | Purpose |
|---|---|
| `src/recommender.py` | Original Module 3 scoring logic — unchanged |
| `src/rag_retriever.py` | TF-IDF inverted index retriever over `music_kb/` |
| `src/llm_client.py` | Gemini 2.5 Flash client — few-shot + baseline prompts |
| `src/agent.py` | 5-step agentic orchestrator with logging |
| `src/confidence.py` | Confidence scoring and guardrail logic |
| `music_kb/*.md` | Genre, mood, energy, and artist knowledge base |
| `app.py` | Streamlit UI — RAG, Baseline, and Fine-Tuning modes |
| `evaluation/test_harness.py` | Automated test harness — 6 profiles, pass/fail |
| `evaluation/retrieval_eval.py` | Single-source vs. multi-source retrieval comparison |
| `model_card.md` | AI collaboration, bias analysis, ethics reflection |
| `logs/agent_run.log` | Auto-generated agent run logs |
