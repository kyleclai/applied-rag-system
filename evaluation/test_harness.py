"""Test harness: runs 6 predefined profiles through the RAG pipeline and prints pass/fail."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.agent import RecommendationAgent

# Try to initialize LLM; degrade gracefully if key is missing
_llm = None
if os.getenv("GEMINI_API_KEY"):
    try:
        from src.llm_client import MusicLLMClient
        _llm = MusicLLMClient()
    except Exception as e:
        print(f"[WARN] LLM unavailable: {e}")

TEST_CASES = [
    {
        "name": "Happy Pop",
        "profile": {"genre": "pop", "mood": "happy", "energy": 0.8},
        "expected_genre": "pop",
    },
    {
        "name": "Chill Lofi",
        "profile": {"genre": "lofi", "mood": "chill", "energy": 0.38},
        "expected_genre": "lofi",
    },
    {
        "name": "Intense Rock",
        "profile": {"genre": "rock", "mood": "intense", "energy": 0.9},
        "expected_genre": "rock",
    },
    {
        "name": "Hip-Hop Confident",
        "profile": {"genre": "hip-hop", "mood": "confident", "energy": 0.78},
        "expected_genre": "hip-hop",
    },
    {
        "name": "Jazz Relaxed",
        "profile": {"genre": "jazz", "mood": "relaxed", "energy": 0.37},
        "expected_genre": "jazz",
    },
    {
        "name": "Metal Aggressive",
        "profile": {"genre": "metal", "mood": "aggressive", "energy": 0.97},
        "expected_genre": "metal",
    },
]

CHECK_LABELS = [
    "genre_match",
    "confidence_ok",
    "explanation_ok",
    "steps_ok",
]


def run_test_harness():
    agent = RecommendationAgent(llm_client=_llm)
    records = []

    print("\n" + "=" * 65)
    print("TEST HARNESS — VibeFinder RAG System")
    print("=" * 65)

    for tc in TEST_CASES:
        try:
            result = agent.run(tc["profile"])

            top_genre = result.top_songs[0]["genre"] if result.top_songs else ""
            genre_match = top_genre == tc["expected_genre"]
            confidence_ok = result.confidence >= 0.5
            exp_lower = result.explanation.lower()
            explanation_ok = (
                len(result.explanation) > 0
                and "i do not know" not in exp_lower
                and "i don't know" not in exp_lower
            )
            steps_ok = len(result.steps) == 5
            passed = genre_match and confidence_ok and explanation_ok and steps_ok

            records.append({
                "name": tc["name"],
                "passed": passed,
                "confidence": result.confidence,
                "genre_match": genre_match,
                "confidence_ok": confidence_ok,
                "explanation_ok": explanation_ok,
                "steps_ok": steps_ok,
            })

            status = "PASS" if passed else "FAIL"
            print(f"\n[{status}] {tc['name']}")
            print(f"  Genre match  : {'✓' if genre_match else '✗'}  (got '{top_genre}', expected '{tc['expected_genre']}')")
            print(f"  Confidence   : {'✓' if confidence_ok else '✗'}  ({result.confidence:.2f})")
            print(f"  Explanation  : {'✓' if explanation_ok else '✗'}")
            print(f"  Agent steps  : {'✓' if steps_ok else '✗'}  ({len(result.steps)}/5)")

        except Exception as e:
            records.append({"name": tc["name"], "passed": False, "confidence": 0.0})
            print(f"\n[FAIL] {tc['name']} — ERROR: {e}")

    # ── Summary ──────────────────────────────────────────────────────────────
    passed_count = sum(1 for r in records if r["passed"])
    print("\n" + "=" * 65)
    print(f"RESULTS: {passed_count}/{len(TEST_CASES)} tests passed")

    print(f"\n  {'Profile':<25} {'Confidence':>12}  {'Status':>6}")
    print(f"  {'-'*25} {'-'*12}  {'-'*6}")
    for r in records:
        conf_str = f"{r['confidence']:.2f}" if "confidence" in r else "ERROR"
        status_str = "PASS" if r["passed"] else "FAIL"
        print(f"  {r['name']:<25} {conf_str:>12}  {status_str:>6}")

    valid_conf = [r["confidence"] for r in records if isinstance(r.get("confidence"), float)]
    if valid_conf:
        print(f"\n  Average confidence: {sum(valid_conf) / len(valid_conf):.2f}")
    print("=" * 65)

    return passed_count, len(TEST_CASES)


if __name__ == "__main__":
    run_test_harness()
