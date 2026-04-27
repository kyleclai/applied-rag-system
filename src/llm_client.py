import os
from typing import List, Tuple

import google.generativeai as genai

GEMINI_MODEL_NAME = "gemini-2.5-flash"

# Few-shot examples that define the "music journalist" tone and grounding behavior
FEW_SHOT_EXAMPLES = """
--- FEW-SHOT EXAMPLES (follow this style) ---

Example 1:
User profile: Genre: jazz | Mood: relaxed | Energy: 0.37
Top songs: Coffee Shop Stories (jazz/relaxed/energy 0.37), Adagio in Grey (classical/melancholic/energy 0.22)
Context from knowledge base: Jazz is characterized by improvisation, complex harmonies, and a warm, intimate feel... Coffee Shop Stories by Slow Stereo is a warm, conversational instrumental track at 0.37 energy...
Explanation: For a listener craving low-key jazz warmth, Coffee Shop Stories is a near-perfect fit. Jazz thrives on its conversational quality — instruments responding to each other in real time — and Slow Stereo captures exactly that laid-back intimacy at 0.37 energy. Adagio in Grey stretches slightly outside your genre preference into classical territory, but its low energy and meditative character make it a natural companion for a relaxed evening.

Example 2:
User profile: Genre: pop | Mood: happy | Energy: 0.8
Top songs: Sunrise City (pop/happy/energy 0.82), Gym Hero (pop/intense/energy 0.93)
Context from knowledge base: Pop music prioritizes accessible melodies, polished production, and broad emotional appeal... Neon Echo is a versatile producer whose pop work features high valence and catchy hooks...
Explanation: Sunrise City is practically purpose-built for your taste — Neon Echo delivers pop's signature melodic clarity with irresistible happy energy at 0.82. The production feels current without being gimmicky. Gym Hero lands in second place: it shares your genre but trades some of that happiness for intensity, pushing the energy to 0.93 — perfect if your mood shifts toward wanting to push harder.

--- END FEW-SHOT EXAMPLES ---
"""


class MusicLLMClient:
    """Gemini client for generating grounded music recommendation explanations."""

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable is not set. "
                "Copy .env.example to .env and add your key."
            )
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(GEMINI_MODEL_NAME)

    def generate_explanation(
        self,
        user_profile: dict,
        top_songs: list,
        retrieved_chunks: List[Tuple[str, str]],
        use_few_shot: bool = True,
    ) -> str:
        """Generate a grounded, music-journalist-style explanation using retrieved context."""
        if not retrieved_chunks:
            return "I do not have enough context to generate a detailed explanation for this profile."

        chunks_text = "\n\n".join(
            f"[{source}]\n{chunk}" for source, chunk in retrieved_chunks
        )
        songs_text = "\n".join(
            f"- {s['title']} by {s['artist']} ({s['genre']}/{s['mood']}, energy {s['energy']})"
            for s in top_songs[:5]
        )
        profile_text = (
            f"Genre: {user_profile.get('genre', 'any')} | "
            f"Mood: {user_profile.get('mood', 'any')} | "
            f"Energy: {user_profile.get('energy', 'any')}"
        )

        few_shot_block = FEW_SHOT_EXAMPLES if use_few_shot else ""

        prompt = f"""You are a music journalist writing personalized listening recommendations.
Use ONLY the retrieved knowledge context below and the song metadata to write your explanation.
Do not invent facts about songs or artists not mentioned in the context.
Write 2-3 sentences per top recommendation in the voice of a knowledgeable music journalist.
If the context does not cover a song well, acknowledge that briefly rather than guessing.

{few_shot_block}
--- RETRIEVED KNOWLEDGE CONTEXT ---
{chunks_text}

--- USER PROFILE ---
{profile_text}

--- TOP RECOMMENDED SONGS (in order) ---
{songs_text}

Write a personalized explanation for why these songs match this listener's taste, grounded in the retrieved context."""

        response = self.model.generate_content(prompt)
        return response.text.strip()

    def generate_explanation_baseline(self, user_profile: dict, top_songs: list) -> str:
        """Baseline prompt: no few-shot examples, no retrieved context. For comparison."""
        songs_text = "\n".join(
            f"- {s['title']} by {s['artist']} ({s['genre']}/{s['mood']}, energy {s['energy']})"
            for s in top_songs[:5]
        )
        profile_text = (
            f"Genre: {user_profile.get('genre', 'any')} | "
            f"Mood: {user_profile.get('mood', 'any')} | "
            f"Energy: {user_profile.get('energy', 'any')}"
        )
        prompt = f"""You are a music recommendation assistant.
Explain why these songs match this listener's taste.

USER PROFILE: {profile_text}
TOP SONGS:
{songs_text}

Write a brief explanation."""
        response = self.model.generate_content(prompt)
        return response.text.strip()

    def rate_confidence(
        self, explanation: str, retrieved_chunks: List[Tuple[str, str]]
    ) -> float:
        """Ask the model to self-rate how grounded the explanation is (0.0–1.0)."""
        if not retrieved_chunks or not explanation:
            return 0.0
        chunks_text = "\n".join(chunk[:300] for _, chunk in retrieved_chunks[:2])
        prompt = f"""Rate how well this music recommendation explanation is supported by the provided context.
Score 0.0 to 1.0:
  1.0 = fully grounded, every claim traceable to the context
  0.5 = partially grounded, some unsupported claims
  0.0 = not grounded, mostly invented facts

Context:
{chunks_text}

Explanation:
{explanation[:400]}

Respond with ONLY a decimal number between 0.0 and 1.0, nothing else."""
        try:
            response = self.model.generate_content(prompt)
            return max(0.0, min(1.0, float(response.text.strip())))
        except (ValueError, Exception):
            return 0.5
