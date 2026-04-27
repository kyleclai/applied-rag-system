from typing import List, Tuple, Optional


FALLBACK_MESSAGE = (
    "The system could not generate a high-confidence explanation for this profile. "
    "The knowledge base may have limited coverage for this genre/mood combination. "
    "Try adjusting your genre or mood preferences, or check that the music_kb/ "
    "folder contains the expected knowledge files."
)


def score_confidence(
    retrieved_chunks: List[Tuple[str, str]],
    llm_response: str,
    top_songs: list,
    llm_rating: Optional[float] = None,
) -> float:
    """Blend three signals into a final confidence score 0.0–1.0.

    Signal 1 (0.0–0.4): retrieval coverage — how many chunks were retrieved
    Signal 2 (0.0–0.4): LLM self-rating of groundedness
    Signal 3 (0.0–0.2): response completeness — non-empty and substantive
    Penalty: "I do not know" detected in response
    """
    # Signal 1: retrieval coverage
    if retrieved_chunks:
        retrieval_signal = min(len(retrieved_chunks) / 4.0, 1.0) * 0.4
    else:
        retrieval_signal = 0.0

    # Signal 2: LLM self-rating
    if llm_rating is not None:
        llm_signal = float(llm_rating) * 0.4
    else:
        llm_signal = 0.2  # neutral default when LLM unavailable

    # Signal 3: response completeness
    if llm_response and len(llm_response) > 100:
        completeness_signal = 0.2
    elif llm_response and len(llm_response) > 30:
        completeness_signal = 0.1
    else:
        completeness_signal = 0.0

    # Penalty: explicit uncertainty in response
    if "i do not know" in llm_response.lower() or "i don't know" in llm_response.lower():
        return 0.2

    total = retrieval_signal + llm_signal + completeness_signal
    return round(min(max(total, 0.0), 1.0), 2)
