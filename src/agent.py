import os
import sys
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional

from src.recommender import load_songs, recommend_songs
from src.rag_retriever import MusicKnowledgeRetriever
from src.confidence import score_confidence, FALLBACK_MESSAGE

os.makedirs("logs", exist_ok=True)

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler("logs/agent_run.log")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False


@dataclass
class AgentResult:
    user_profile: dict
    search_terms: List[str] = field(default_factory=list)
    retrieved_chunks: List[Tuple[str, str]] = field(default_factory=list)
    top_songs: list = field(default_factory=list)
    scored_results: list = field(default_factory=list)  # (song, score, reasons) tuples
    explanation: str = ""
    confidence: float = 0.0
    guardrail_triggered: bool = False
    steps: List[Dict[str, Any]] = field(default_factory=list)


class RecommendationAgent:
    """5-step agentic pipeline for RAG-enhanced music recommendations.

    Observable steps:
      1. Profile Analysis  — extract search terms from user profile
      2. Knowledge Retrieval — retrieve KB chunks via inverted index
      3. Song Scoring       — score catalog with existing recommender logic
      4. LLM Generation    — generate explanation grounded in retrieved context
      5. Confidence Check  — score confidence and apply guardrail if needed
    """

    def __init__(
        self,
        songs_path: str = "data/songs.csv",
        kb_folder: str = "music_kb",
        llm_client=None,
    ):
        self.songs = load_songs(songs_path)
        self.retriever = MusicKnowledgeRetriever(kb_folder=kb_folder)
        self.llm = llm_client

    def run(self, user_profile: dict, mode: str = "balanced") -> AgentResult:
        result = AgentResult(user_profile=user_profile)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"=== Agent run started {ts} | profile={user_profile} ===")

        # ── Step 1: Profile Analysis ──────────────────────────────────────────
        logger.info("Step 1 [Profile Analysis] — extracting search terms from profile")
        search_terms = self._extract_search_terms(user_profile)
        result.search_terms = search_terms
        result.steps.append({
            "step": 1,
            "name": "Profile Analysis",
            "input": str(user_profile),
            "output": f"Search terms extracted: {search_terms}",
        })
        logger.info(f"  → Terms: {search_terms}")

        # ── Step 2: Knowledge Retrieval ───────────────────────────────────────
        query = " ".join(search_terms)
        logger.info(f"Step 2 [Knowledge Retrieval] — query: '{query}'")
        chunks = self.retriever.retrieve(query, top_k=4)
        result.retrieved_chunks = chunks
        sources = sorted({src for src, _ in chunks})
        result.steps.append({
            "step": 2,
            "name": "Knowledge Retrieval",
            "input": f"Query: '{query}'",
            "output": f"Retrieved {len(chunks)} chunks from: {sources if sources else 'no matching documents'}",
        })
        logger.info(f"  → {len(chunks)} chunks from {sources}")

        # ── Step 3: Song Scoring ──────────────────────────────────────────────
        logger.info(f"Step 3 [Song Scoring] — scoring {len(self.songs)} songs in '{mode}' mode")
        scored = recommend_songs(user_profile, self.songs, k=5, mode=mode)
        result.top_songs = [song for song, _score, _reasons in scored]
        result.scored_results = scored
        top_label = f"{scored[0][0]['title']} ({scored[0][1]:.2f})" if scored else "none"
        result.steps.append({
            "step": 3,
            "name": "Song Scoring",
            "input": f"mode={mode} | genre={user_profile.get('genre')} | mood={user_profile.get('mood')}",
            "output": f"Top song: {top_label}. All 5 selected.",
        })
        logger.info(f"  → Top song: {top_label}")

        # ── Step 4: LLM Generation ────────────────────────────────────────────
        if self.llm is not None:
            logger.info("Step 4 [LLM Generation] — generating few-shot explanation")
            explanation = self.llm.generate_explanation(user_profile, result.top_songs, chunks)
            llm_rating = self.llm.rate_confidence(explanation, chunks)
            logger.info(f"  → {len(explanation)} chars generated, LLM self-rating: {llm_rating:.2f}")
        else:
            explanation = "LLM explanation unavailable (GEMINI_API_KEY not set)."
            llm_rating = None
            logger.info("Step 4 [LLM Generation] — skipped (no LLM client)")

        result.explanation = explanation
        result.steps.append({
            "step": 4,
            "name": "LLM Generation",
            "input": f"{len(chunks)} retrieved chunks + {len(result.top_songs)} top songs",
            "output": (explanation[:120] + "…") if len(explanation) > 120 else explanation,
        })

        # ── Step 5: Confidence Assessment ────────────────────────────────────
        logger.info("Step 5 [Confidence Assessment] — computing confidence score")
        conf = score_confidence(chunks, explanation, result.top_songs, llm_rating)
        result.confidence = conf
        guardrail = conf < 0.4
        result.guardrail_triggered = guardrail
        if guardrail:
            result.explanation = FALLBACK_MESSAGE
            logger.warning(f"  → Guardrail TRIGGERED (confidence {conf:.2f} < 0.4)")
        else:
            logger.info(f"  → Confidence: {conf:.2f} | Guardrail PASSED")

        result.steps.append({
            "step": 5,
            "name": "Confidence Assessment",
            "input": f"{len(chunks)} chunks | explanation length {len(explanation)} chars",
            "output": f"Confidence: {conf:.2f} | Guardrail: {'TRIGGERED ⚠️' if guardrail else 'PASSED ✓'}",
        })

        logger.info(f"=== Run complete. Confidence: {conf:.2f} ===\n")
        return result

    def _extract_search_terms(self, user_profile: dict) -> List[str]:
        terms = []
        for key in ("genre", "mood"):
            val = user_profile.get(key)
            if val:
                terms.append(str(val))
        energy = user_profile.get("energy")
        if energy is not None:
            e = float(energy)
            if e < 0.4:
                terms.append("low energy")
            elif e > 0.7:
                terms.append("high energy")
            else:
                terms.append("medium energy")
        return terms
