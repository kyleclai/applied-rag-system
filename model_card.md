# Model Card — VibeFinder RAG

## 1. Model Name and Version

**VibeFinder RAG v1.0**  
Extended from: VibeFinder 1.0 (Module 3 — Music Recommender Simulation)  
LLM backbone: Gemini 2.5 Flash (`gemini-2.5-flash`) via Google Generative AI API  
Retrieval: TF-IDF inverted index over a 4-document music knowledge base

---

## 2. Intended Use

VibeFinder RAG is a music recommendation system designed for individual listeners who want both ranked recommendations AND grounded natural-language explanations for why those recommendations fit their taste.

**Primary users:** Individual music listeners exploring a curated catalog  
**Use context:** Personal listening assistance, music discovery  
**Out of scope:** Commercial music platform deployment, real-time streaming catalog access, personalization based on listening history

---

## 3. How It Works

The system runs a 5-step pipeline:

1. **Profile Analysis** — Extracts genre, mood, and energy-level search terms from the user's taste profile
2. **Knowledge Retrieval** — Uses an inverted word index to find relevant chunks from `music_kb/` documents (genres, moods, energy profiles, artist context)
3. **Song Scoring** — Applies the original Module 3 weighted scoring algorithm: genre match (+2.0 balanced), mood match (+1.0), energy proximity (continuous)
4. **LLM Generation** — Gemini 2.5 Flash generates an explanation using only the retrieved chunks and song metadata; few-shot examples constrain the output to a "music journalist" style
5. **Confidence Assessment** — Blends retrieval coverage, LLM self-rating, and response completeness into a 0.0–1.0 score; triggers a guardrail fallback if confidence < 0.4

**Modes:**
- **RAG-Enhanced** — Full pipeline with LLM explanations
- **Baseline** — Original Module 3 scoring only (no retrieval, no LLM)
- **Fine-Tuning Comparison** — Side-by-side few-shot vs. baseline LLM output

---

## 4. Data

**Catalog:** 20 songs with 12 attributes each (`data/songs.csv`)  
Attributes: id, title, artist, genre, mood, energy, tempo_bpm, valence, danceability, acousticness, popularity, release_decade

**Knowledge Base:** 4 manually authored `.md` files in `music_kb/`:
- `genres.md` — Descriptions of 17 genres in the catalog
- `moods.md` — Descriptions of 16 mood categories
- `energy_profiles.md` — What low/medium/high energy levels feel like
- `artist_context.md` — Brief notes on 18 artists

All knowledge base content was authored specifically for this system. No external datasets were used for retrieval. All songs are fictional.

---

## 5. Strengths

- **Grounded explanations:** The RAG system constrains the LLM to use only retrieved content, significantly reducing hallucination risk compared to free-form generation
- **Transparent pipeline:** All 5 agent steps are logged and visible in the UI — the system's reasoning is observable
- **Preserves original logic:** The Module 3 scoring algorithm is unchanged; RAG adds explanation without altering ranking
- **Graceful degradation:** Works without an API key (no LLM, but scoring and retrieval still function)
- **Measurable improvement:** The retrieval evaluation quantifies the benefit of multi-source KB over single-source

---

## 6. Limitations and Bias

### Sparse catalog
With only 1–2 songs per genre, genre match almost guarantees a top-3 finish regardless of mood or energy. Users asking for "metal" will get one good match and four unrelated ones. The RAG explanation layer can help contextualize why the top result dominates, but it cannot fix catalog sparsity.

### Genre dominance in scoring
In `balanced` mode, the genre weight (+2.0) is double the mood weight (+1.0). This means a wrong-genre but correct-mood song can rank below a correct-genre but wrong-mood song. The `mood-first` scoring mode addresses this, but the tradeoff shifts rather than disappears.

### Keyword retrieval blind spots
The TF-IDF inverted index cannot handle semantic gaps. If a user describes wanting "melancholic late-night music" and the knowledge base uses "moody" or "dreamy" instead, retrieval may miss relevant chunks. The knowledge base was written to minimize this, but edge cases remain.

### Confidence score reliability
The confidence score is a proxy, not a guarantee. A high confidence (0.70+) means the explanation is *grounded* — not that it is *correct*. An explanation grounded in retrieved facts can still be shallow or musically debatable.

### Fixed knowledge base
The knowledge base was hand-authored for this catalog. Adding new songs or genres requires manually updating KB files — there is no automated ingestion pipeline.

### No temporal adaptation
The system has no memory — identical profiles always produce identical results. It cannot learn from user feedback or adapt to preference drift over time.

---

## 7. Evaluation Results

### Unit tests
```
pytest tests/  →  2/2 PASSED  (original recommender logic unchanged)
```

### Test harness (6 predefined profiles)
```
6/6 tests PASSED
Average confidence: 0.70
All profiles: correct genre match, confidence ≥ 0.5, non-empty explanations, 5/5 steps complete
```

### Retrieval evaluation
```
Single-source (genres.md):   hit rate 100%  avg coverage 0.46
Multi-source (all 4 files):  hit rate 75%   avg coverage 0.54
```
Multi-source retrieval improves average coverage by +17%, meaning it retrieves more of the expected documents when it finds the right ones. The lower hit rate with multi-source is a real tradeoff: more competing documents can displace expected files from the top-3 retrieval window — a genuine RAG design limitation to improve on.

### Confidence scoring behavior
Without an LLM (API key not set): confidence defaults to ~0.70 (high retrieval coverage + neutral LLM signal + complete explanation message). With an LLM: confidence reflects actual LLM self-rating and may vary between 0.4–0.95 depending on profile-KB alignment.

---

## 8. Future Work

- **Embedding-based retrieval:** Replace the inverted index with vector similarity to handle semantic gaps
- **Catalog expansion:** Add 50–100 songs per genre for genuine within-genre diversity
- **User feedback loop:** Let users rate explanations; use ratings to improve retrieval weights
- **Multi-turn conversation:** Support follow-up questions ("Why is this ranked above X?")
- **Diversity penalty:** Penalize same-genre/artist clustering in top-K results

---

## 9. Reflection

### AI Collaboration During This Project

**Helpful suggestion:** When designing the confidence scoring function, the AI (Claude) suggested splitting the signal into three independent components (retrieval coverage, LLM self-rating, response completeness) rather than using a single heuristic. This modular approach made it much easier to tune thresholds and debug cases where confidence was unexpectedly low or high. The separation of concerns made the scoring transparent and auditable.

**Flawed suggestion:** The AI initially suggested using `logging.basicConfig()` at the module level in `agent.py`, which can add duplicate log handlers when modules are reloaded — a subtle Python logging issue common in Streamlit environments. The fix was to verify handler state before configuration, which the initial suggestion overlooked.

### Limitations and Biases Revisited

The most significant bias in VibeFinder RAG is inherited from the catalog: sparse genre representation means the scoring algorithm and the retrieval system both over-reward genre match. A user asking for "metal" essentially has one good option. The RAG explanation layer can contextualize this, but cannot compensate for it.

The knowledge base introduces a second bias: it was authored by one person, meaning genre and mood descriptions reflect a particular perspective on music. The KB is a simplification designed for retrieval quality, not a musicological authority.

### Could This System Be Misused?

The primary risk is over-trust: treating AI explanations as authoritative musical expertise. The confidence score and source citations exist to make limitations visible, but users may not read them. The few-shot prompting makes outputs sound authoritative even when KB coverage is thin — the guardrail threshold is conservative by design to avoid blocking most legitimate use.

### What Surprised Me

The most surprising finding was that multi-source retrieval can decrease hit rate while increasing coverage quality. Intuitively, more documents should always improve retrieval. In practice, more competing documents means the retrieval window fills faster with off-target matches, potentially displacing directly relevant files. This counterintuitive result is the clearest evidence that RAG quality is not just about having more data — it's about how retrieval mechanisms allocate attention across that data.
