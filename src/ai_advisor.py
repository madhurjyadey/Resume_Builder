"""
ai_advisor.py
-------------
Uses Google's Gemini API to analyze a resume against a job description
and generate concrete, actionable suggestions for what to change.

This is a *qualitative* layer on top of the quantitative scoring in
matcher.py: the embedding/skill-matching pipeline tells you WHAT is
missing, and this module asks an LLM to explain WHY it matters and HOW
to address it in the resume's actual wording.

Setup:
    1. Get a free API key from https://aistudio.google.com/apikey
    2. Set it as an environment variable:
           export GEMINI_API_KEY="your-key-here"
       ...or pass it directly to GeminiAdvisor(api_key="...")

Docs: https://ai.google.dev/gemini-api/docs
"""

import os
import json
from dataclasses import dataclass, field
from typing import List, Optional

from google import genai
from google.genai import types

DEFAULT_MODEL = "gemini-2.5-flash"  # best price/performance as of mid-2026
# Swap to "gemini-3.5-flash" for higher-quality (slower/costlier) suggestions.


@dataclass
class AISuggestions:
    key_changes: List[str] = field(default_factory=list)
    bullet_rewrites: List[dict] = field(default_factory=list)  # {"original": ..., "suggested": ...}
    summary: str = ""
    raw_response: str = ""


class GeminiAdvisor:
    """Wraps the Gemini API to turn resume/JD text + skill gaps into
    concrete, editable suggestions."""

    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL):
        resolved_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "No Gemini API key found. Pass api_key=... or set the "
                "GEMINI_API_KEY environment variable. Get a free key at "
                "https://aistudio.google.com/apikey"
            )
        self.client = genai.Client(api_key=resolved_key)
        self.model = model

    def _build_prompt(
        self,
        resume_text: str,
        jd_text: str,
        matched_skills: List[str],
        missing_skills: List[str],
    ) -> str:
        return f"""You are an expert technical resume reviewer and career coach.

You are given a candidate's RESUME and a JOB DESCRIPTION, plus a pre-computed
list of skills that already match and skills that appear to be missing.

Your job: identify the most impactful, honest changes the candidate could
make to their resume to better align with this specific job description.

RULES:
- Only suggest emphasizing or rewording experience the candidate actually
  has evidence of in their resume. NEVER invent skills or experience.
- If a missing skill genuinely isn't reflected anywhere in the resume, say
  so plainly rather than suggesting the candidate fabricate it.
- Be specific and concrete, not generic ("use more action verbs" is not useful).
- Prioritize by impact: what would most improve this specific match.

RESUME:
---
{resume_text[:6000]}
---

JOB DESCRIPTION:
---
{jd_text[:4000]}
---

ALREADY MATCHED SKILLS: {", ".join(matched_skills) if matched_skills else "None detected"}
SKILLS THE JD WANTS BUT THE RESUME DOESN'T CLEARLY SHOW: {", ".join(missing_skills) if missing_skills else "None"}

Respond with ONLY valid JSON (no markdown fences, no preamble), in exactly
this shape:

{{
  "summary": "2-3 sentence overview of the biggest gap and opportunity",
  "key_changes": [
    "Specific, actionable change #1",
    "Specific, actionable change #2",
    "... 4-6 total, ordered by impact"
  ],
  "bullet_rewrites": [
    {{
      "original": "an actual bullet point copied from the resume that could be improved",
      "suggested": "a rewritten version that better aligns with the JD, using only truthful claims"
    }}
  ]
}}

Include 2-4 bullet_rewrites, only for bullets that genuinely exist in the
resume text above. If you can't find suitable bullets to rewrite, return an
empty list for bullet_rewrites rather than inventing one.
"""

    def generate_suggestions(
        self,
        resume_text: str,
        jd_text: str,
        matched_skills: List[str],
        missing_skills: List[str],
    ) -> AISuggestions:
        prompt = self._build_prompt(resume_text, jd_text, matched_skills, missing_skills)

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,  # low temperature: stay grounded, avoid embellishment
                response_mime_type="application/json",
            ),
        )

        raw_text = response.text or ""

        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            # Fallback: model didn't return clean JSON. Surface the raw text
            # so the caller can still show *something* useful.
            return AISuggestions(
                summary="Could not parse a structured response from Gemini.",
                key_changes=[],
                bullet_rewrites=[],
                raw_response=raw_text,
            )

        return AISuggestions(
            summary=parsed.get("summary", ""),
            key_changes=parsed.get("key_changes", []),
            bullet_rewrites=parsed.get("bullet_rewrites", []),
            raw_response=raw_text,
        )