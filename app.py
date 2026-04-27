import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.recommender import load_songs, recommend_songs, SCORING_MODES
from src.agent import RecommendationAgent

try:
    from src.llm_client import MusicLLMClient
    _HAS_LLM_CLASS = True
except ImportError:
    _HAS_LLM_CLASS = False

st.set_page_config(page_title="VibeFinder RAG", page_icon="🎵", layout="wide")
st.title("🎵 VibeFinder RAG")
st.caption(
    "Module 3 VibeFinder 1.0 extended with Retrieval-Augmented Generation — "
    "recommendations grounded in a music knowledge base."
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Your Taste Profile")
    genre = st.selectbox("Favorite Genre", [
        "pop", "lofi", "rock", "ambient", "jazz", "synthwave", "indie pop",
        "hip-hop", "edm", "country", "classical", "r&b", "folk", "metal",
        "blues", "reggae", "electronic",
    ])
    mood = st.selectbox("Favorite Mood", [
        "happy", "chill", "intense", "relaxed", "moody", "focused",
        "confident", "energetic", "nostalgic", "melancholic", "romantic",
        "peaceful", "aggressive", "soulful", "uplifting", "dreamy",
    ])
    energy = st.slider("Target Energy (0 = very calm, 1 = very intense)", 0.0, 1.0, 0.7, 0.05)

    st.divider()
    scoring_mode = st.radio("Scoring Mode", list(SCORING_MODES.keys()), index=0)

    st.divider()
    app_mode = st.radio("System Mode", ["RAG-Enhanced", "Baseline (original scoring)", "Fine-Tuning Comparison"])
    if app_mode == "Fine-Tuning Comparison":
        st.caption("Shows side-by-side: few-shot (specialized) vs. baseline (generic) LLM output.")

    run = st.button("Get Recommendations", type="primary", use_container_width=True)

user_profile = {"genre": genre, "mood": mood, "energy": energy}

# ── Baseline mode ─────────────────────────────────────────────────────────────
if run and app_mode == "Baseline (original scoring)":
    st.subheader("Baseline Recommendations")
    st.caption("VibeFinder 1.0 — no retrieval, no LLM, pure scoring")
    songs = load_songs("data/songs.csv")
    results = recommend_songs(user_profile, songs, k=5, mode=scoring_mode)
    for i, (song, score, reasons) in enumerate(results, 1):
        with st.container(border=True):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**#{i} {song['title']}** — *{song['artist']}*")
                st.caption(f"{song['genre']} · {song['mood']} · energy {song['energy']}")
            with col2:
                st.metric("Score", f"{score:.2f}")
            st.caption(f"Scoring: {reasons}")

# ── RAG-Enhanced mode ─────────────────────────────────────────────────────────
elif run:
    st.subheader("RAG-Enhanced Recommendations")

    # Initialize LLM if available
    llm_client = None
    if _HAS_LLM_CLASS and os.getenv("GEMINI_API_KEY"):
        try:
            llm_client = MusicLLMClient()
        except RuntimeError as e:
            st.warning(f"LLM unavailable: {e}")
    elif not os.getenv("GEMINI_API_KEY"):
        st.info("Add GEMINI_API_KEY to .env to enable AI-generated explanations.")

    agent = RecommendationAgent(llm_client=llm_client)

    with st.spinner("Running 5-step RAG pipeline…"):
        result = agent.run(user_profile, mode=scoring_mode)

    # ── Agent steps ──────────────────────────────────────────────────────────
    st.subheader("Agent Pipeline — Observable Steps")
    for step in result.steps:
        icon = "✅" if "TRIGGERED" not in step["output"] else "⚠️"
        with st.expander(f"{icon} Step {step['step']}: {step['name']}"):
            st.markdown(f"**Input:** `{step['input']}`")
            st.markdown(f"**Output:** {step['output']}")

    st.divider()

    # ── Confidence badge ─────────────────────────────────────────────────────
    col_conf, col_guard = st.columns([1, 3])
    with col_conf:
        st.metric("Confidence Score", f"{result.confidence:.2f}")
    with col_guard:
        if result.guardrail_triggered:
            st.error("⚠️ Guardrail triggered — confidence below 0.4 threshold.")
        else:
            st.success("✓ Guardrail passed — explanation is sufficiently grounded.")

    st.divider()

    # ── Top songs ────────────────────────────────────────────────────────────
    st.subheader("Top 5 Songs")
    for i, (song, score, reasons) in enumerate(result.scored_results, 1):
        with st.container(border=True):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**#{i} {song['title']}** — *{song['artist']}*")
                st.caption(f"{song['genre']} · {song['mood']} · energy {song['energy']}")
            with col2:
                st.metric("Score", f"{score:.2f}")
            st.caption(f"Scoring: {reasons}")

    # ── AI Explanation ───────────────────────────────────────────────────────
    st.subheader("AI Explanation")
    if result.guardrail_triggered:
        st.warning(result.explanation)
    else:
        st.markdown(result.explanation)

    # Retrieved sources
    if result.retrieved_chunks:
        with st.expander("📚 Retrieved Knowledge Sources"):
            for source, chunk in result.retrieved_chunks:
                st.markdown(f"**`{source}`**")
                st.text(chunk[:400] + ("…" if len(chunk) > 400 else ""))
                st.divider()

# ── Fine-Tuning Comparison mode ───────────────────────────────────────────────
elif run and app_mode == "Fine-Tuning Comparison":
    st.subheader("Fine-Tuning: Few-Shot vs. Baseline Comparison")
    st.caption(
        "Demonstrates specialization via few-shot prompting. "
        "The same songs are explained two ways: with journalist-style examples (few-shot) "
        "and without any style guidance (baseline)."
    )

    llm_client = None
    if _HAS_LLM_CLASS and os.getenv("GEMINI_API_KEY"):
        try:
            llm_client = MusicLLMClient()
        except RuntimeError as e:
            st.error(f"LLM unavailable: {e}")
            st.stop()
    else:
        st.warning("Set GEMINI_API_KEY in .env to run this comparison.")
        st.stop()

    from src.rag_retriever import MusicKnowledgeRetriever
    retriever = MusicKnowledgeRetriever()
    songs = load_songs("data/songs.csv")
    scored = recommend_songs(user_profile, songs, k=5, mode=scoring_mode)
    top_songs = [s for s, _, _ in scored]

    query = f"{user_profile['genre']} {user_profile['mood']}"
    chunks = retriever.retrieve(query, top_k=4)

    with st.spinner("Generating both explanations…"):
        few_shot_output = llm_client.generate_explanation(
            user_profile, top_songs, chunks, use_few_shot=True
        )
        baseline_output = llm_client.generate_explanation_baseline(user_profile, top_songs)

    col_fs, col_bl = st.columns(2)
    with col_fs:
        st.markdown("#### Few-Shot (Music Journalist Style)")
        st.info(few_shot_output)
        st.caption("Uses 2-3 journalist-style examples + retrieved KB context to shape tone and grounding.")
    with col_bl:
        st.markdown("#### Baseline (Generic Prompt)")
        st.info(baseline_output)
        st.caption("Generic prompt with no examples and no retrieved context.")

    st.divider()
    st.markdown("**What differs:** Few-shot output uses specific retrieved facts and a consistent journalistic voice. "
                "Baseline output is more conversational and generic, without grounding in the knowledge base.")
