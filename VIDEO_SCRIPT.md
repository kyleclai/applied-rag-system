# VibeFinder RAG — Loom Video Script

**Target length:** 6 minutes  
**Goal:** Cover all 4 rubric checkboxes — end-to-end run, AI feature behavior, reliability/guardrail, clear outputs

---

## Before You Hit Record

Have ready:
- Browser with `streamlit run app.py` already loaded and visible
- Terminal window open at the project root, nothing run yet
- Sidebar collapsed so the full app is visible on screen

---

## Segment 1 — Introduction (0:00 – 0:30)

**On screen:** Streamlit app, sidebar visible

**Say:**
> "This is VibeFinder RAG — my final project for CodePath AI110. I started from my Module 3 music recommender, which originally just scored songs with a weighted algorithm, and extended it into a full RAG system. It now retrieves context from a music knowledge base, uses Gemini to write grounded explanations, and runs a 5-step agentic pipeline you can watch in real time. Let me walk through three examples."

---

## Segment 2 — Example 1: Pop / Happy / High Energy (0:30 – 2:30)

**On screen:** Set sidebar to:

| Setting | Value |
|---|---|
| Genre | **pop** |
| Mood | **happy** |
| Energy | **0.8** |
| Scoring Mode | **balanced** |
| System Mode | **RAG-Enhanced** |

Click **Get Recommendations**.

**Say:**
> "First profile — pop, happy, high energy. I'll open each agent step so you can see the intermediate reasoning the system produces."

**Click open Step 1 — pause 2 seconds.**
> "Step 1 broke the profile into search keywords: pop, happy, and high energy."

**Click open Step 2 — pause 2 seconds.**
> "Step 2 queried the knowledge base and retrieved 4 relevant chunks — pulling from artist context, energy profiles, and genre descriptions."

**Click open Step 3 — pause 2 seconds.**
> "Step 3 ran the original VibeFinder scoring algorithm. Sunrise City comes out on top at 3.98 out of 4."

**Click open Step 4 — pause 2 seconds.**
> "Step 4 sent those retrieved chunks and the top songs to Gemini to generate an explanation grounded only in what was retrieved."

**Click open Step 5 — pause 2 seconds.**
> "Step 5 assessed confidence — 0.92. Guardrail passed. The system is confident this explanation is well-supported."

**Scroll down past top 5 songs to AI Explanation.**
> "Here's the actual explanation Gemini produced. Notice it references specific details — Indigo Parade's 2010 sound, the energy profile context — all traceable back to the retrieved chunks."

**Scroll down to Retrieved Knowledge Sources and expand it.**
> "These are the exact chunks Gemini was allowed to use. Nothing outside of these."

---

## Segment 3 — Example 2: Jazz / Relaxed / Low Energy (2:30 – 4:00)

**On screen:** Change sidebar to:

| Setting | Value |
|---|---|
| Genre | **jazz** |
| Mood | **relaxed** |
| Energy | **0.35** |
| Scoring Mode | **balanced** |
| System Mode | **RAG-Enhanced** |

Click **Get Recommendations**.

**Say:**
> "Second profile — jazz, relaxed, low energy. Watch how the retrieved sources change to match this profile."

**Point to Step 2 output.**
> "This time it pulled jazz and low energy context from the knowledge base. Different profile, different retrieval."

**Point to Step 5 output.**
> "Confidence is 1.0 this time — the knowledge base has very strong coverage for jazz and relaxed. The system is fully grounded."

**Scroll to the AI Explanation.**
> "The explanation references Slow Stereo's specific recording style and explains why jazz at this energy level fits a relaxed listening context. Every claim is traceable to what was retrieved."

---

## Segment 4 — Fine-Tuning Comparison (4:00 – 4:45)

**On screen:** Keep jazz / relaxed / 0.35. Change System Mode to **Fine-Tuning Comparison**. Click **Get Recommendations**.

**Say:**
> "This is the specialization stretch feature. Same profile, two different prompts running side by side."

**Point to the left column.**
> "The left uses few-shot examples to constrain Gemini to a music journalist style — specific, grounded, cites retrieved facts."

**Point to the right column.**
> "The right is a generic baseline prompt — no examples, no retrieved context. The output is more conversational and generic."

> "The tone, specificity, and grounding measurably differ between the two. That's the point of the comparison."

---

## Segment 5 — Test Harness (4:45 – 5:45)

**On screen:** Switch to the terminal window at the project root.

**Type and run:**
```bash
python evaluation/test_harness.py
```

**While it runs, say:**
> "This is the reliability component — an automated test harness that runs 6 predefined profiles through the full pipeline and checks three things: does the top song match the expected genre, is confidence above 0.5, and do all 5 agent steps complete without error."

**Let it finish. Point to the results table.**
> "6 out of 6 pass. Average confidence 0.70. This is how I know the system behaves consistently across different profiles — not just on the examples I hand-picked for the demo."

---

## Segment 6 — Close (5:45 – 6:00)

**Say:**
> "Full source code, README, model card, and setup instructions are in the GitHub repo. Thanks for watching."

---

## Quick Reference Card

Keep this visible on a second screen while recording.

| Segment | Genre | Mood | Energy | Scoring Mode | System Mode |
|---|---|---|---|---|---|
| Example 1 | pop | happy | 0.8 | balanced | RAG-Enhanced |
| Example 2 | jazz | relaxed | 0.35 | balanced | RAG-Enhanced |
| Fine-Tuning | jazz | relaxed | 0.35 | balanced | Fine-Tuning Comparison |
| Terminal | — | — | — | — | `python evaluation/test_harness.py` |

---

## Rubric Checkboxes Covered

| Checkbox | Covered in |
|---|---|
| End-to-end system run (2–3 inputs) | Segments 2 and 3 |
| AI feature behavior (RAG, agent steps) | Segments 2 and 3 — expand all 5 steps |
| Reliability / guardrail behavior | Segment 5 — test harness terminal output |
| Clear outputs for each case | Segments 2, 3, and 4 — explanation + confidence visible |
