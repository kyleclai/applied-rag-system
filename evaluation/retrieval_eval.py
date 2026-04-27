"""Retrieval evaluation: compare single-source vs. multi-source knowledge retrieval."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_retriever import MusicKnowledgeRetriever

# Queries and the KB files expected to contain relevant chunks
TEST_QUERIES = [
    ("pop happy high energy", ["genres.md", "moods.md", "energy_profiles.md"]),
    ("jazz relaxed low energy", ["genres.md", "moods.md", "energy_profiles.md"]),
    ("metal aggressive intense", ["genres.md", "moods.md"]),
    ("lofi chill focused", ["genres.md", "moods.md"]),
    ("hip-hop confident medium energy", ["genres.md", "moods.md"]),
    ("ambient dreamy electronic", ["genres.md", "moods.md"]),
    ("Voltline Storm Runner rock", ["artist_context.md", "genres.md"]),
    ("Vienna Strings classical melancholic", ["artist_context.md", "genres.md"]),
]


def evaluate_retrieval(retriever: MusicKnowledgeRetriever, label: str):
    hits = 0
    total_coverage = 0.0
    doc_names = [fname for fname, _ in retriever.documents]
    print(f"\n--- {label} ---")
    print(f"  Documents loaded: {doc_names}")

    for query, expected_sources in TEST_QUERIES:
        chunks = retriever.retrieve(query, top_k=4)
        retrieved_sources = {src for src, _ in chunks}
        hit = any(src in retrieved_sources for src in expected_sources)
        hits += int(hit)
        found = sum(1 for src in expected_sources if src in retrieved_sources)
        coverage = found / len(expected_sources)
        total_coverage += coverage
        status = "HIT " if hit else "MISS"
        print(f"  [{status}] '{query[:40]:<40}' → {sorted(retrieved_sources)} (cov {coverage:.2f})")

    hit_rate = hits / len(TEST_QUERIES)
    avg_coverage = total_coverage / len(TEST_QUERIES)
    print(f"\n  Hit rate    : {hits}/{len(TEST_QUERIES)} ({hit_rate:.0%})")
    print(f"  Avg coverage: {avg_coverage:.2f}")
    return hit_rate, avg_coverage


def run_retrieval_eval():
    print("\n" + "=" * 68)
    print("RETRIEVAL EVALUATION — Single-Source vs. Multi-Source")
    print("=" * 68)

    # Single-source: genres.md only
    single = MusicKnowledgeRetriever(kb_folder="music_kb")
    single.documents = [(f, t) for f, t in single.documents if f == "genres.md"]
    single.index = single.build_index(single.documents)
    single_hit, single_cov = evaluate_retrieval(single, "Single-Source (genres.md only)")

    # Multi-source: all kb files
    multi = MusicKnowledgeRetriever(kb_folder="music_kb")
    multi_hit, multi_cov = evaluate_retrieval(multi, "Multi-Source (all 4 kb files)")

    # Summary table
    print("\n" + "=" * 68)
    print("COMPARISON")
    print(f"  {'Metric':<30} {'Single-Source':>15} {'Multi-Source':>14} {'Δ':>6}")
    print(f"  {'-'*30} {'-'*15} {'-'*14} {'-'*6}")
    print(
        f"  {'Hit Rate':<30} {single_hit:>14.0%} {multi_hit:>13.0%} "
        f"{multi_hit - single_hit:>+5.0%}"
    )
    print(
        f"  {'Avg Coverage Score':<30} {single_cov:>15.2f} {multi_cov:>14.2f} "
        f"{multi_cov - single_cov:>+6.2f}"
    )
    print("=" * 68)

    return {
        "single": {"hit_rate": single_hit, "avg_coverage": single_cov},
        "multi": {"hit_rate": multi_hit, "avg_coverage": multi_cov},
    }


if __name__ == "__main__":
    run_retrieval_eval()
