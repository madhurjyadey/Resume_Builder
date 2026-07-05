"""
matcher.py
----------
Core scoring engine: embeds resume/JD text with Sentence-BERT,
extracts skills via fuzzy keyword matching against a taxonomy,
and combines both signals into a final match report.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz

from src.skills_data import ALL_SKILLS

# Skills at or under this length are matched with strict word-boundary
# regex only (no substring/fuzzy fallback), since short tokens like "r"
# or "go" would otherwise match almost any text.
SHORT_SKILL_LEN_THRESHOLD = 3

# Weights for the final blended score. Tune these based on how much you
# trust "overall semantic similarity" vs. "explicit skill overlap".
SEMANTIC_WEIGHT = 0.5
SKILL_OVERLAP_WEIGHT = 0.5

FUZZY_MATCH_THRESHOLD = 85  # 0-100, higher = stricter matching


@dataclass
class MatchResult:
    overall_score: float
    semantic_similarity: float
    skill_overlap_pct: float
    matched_skills: List[str] = field(default_factory=list)
    missing_skills: List[str] = field(default_factory=list)
    resume_only_skills: List[str] = field(default_factory=list)


class ResumeJobMatcher:
    """
    Loads a Sentence-BERT model once and exposes methods to score a
    resume against a job description.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    # ---------- Skill extraction ----------

    def extract_skills(self, text: str, taxonomy: List[str] = None) -> List[str]:
        """
        Find which skills from the taxonomy appear in `text`.

        - Short skills (e.g. "r", "go", "c#") are matched with a strict
          word-boundary regex only, to avoid false positives like "r"
          matching inside "for" or "experience".
        - Longer / multi-word skills use substring matching plus a fuzzy
          fallback so minor wording differences (plurals, punctuation,
          hyphenation) still count as a match.
        """
        taxonomy = taxonomy or ALL_SKILLS
        text_lower = text.lower()
        found = []

        for skill in taxonomy:
            skill_lower = skill.lower()

            if len(skill_lower) <= SHORT_SKILL_LEN_THRESHOLD:
                pattern = r"(?<![a-z0-9])" + re.escape(skill_lower) + r"(?![a-z0-9])"
                if re.search(pattern, text_lower):
                    found.append(skill)
                continue

            if skill_lower in text_lower:
                found.append(skill)
                continue

            # Fuzzy fallback for near-matches on longer skill phrases
            # (e.g. "Machine-Learning" vs "machine learning")
            score = fuzz.partial_ratio(skill_lower, text_lower)
            if score >= FUZZY_MATCH_THRESHOLD:
                found.append(skill)

        return sorted(set(found))

    # ---------- Semantic similarity ----------

    def semantic_similarity(self, resume_text: str, jd_text: str) -> float:
        """Cosine similarity between whole-document embeddings, scaled to 0-100."""
        embeddings = self.model.encode([resume_text, jd_text], convert_to_numpy=True)
        sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        # Cosine similarity for sentence embeddings is typically in [0, 1] for
        # related text; clip defensively and rescale to a 0-100 score.
        return float(np.clip(sim, 0, 1) * 100)

    # ---------- Full scoring pipeline ----------

    def score(self, resume_text: str, jd_text: str) -> MatchResult:
        resume_skills = self.extract_skills(resume_text)
        jd_skills = self.extract_skills(jd_text)

        matched = sorted(set(resume_skills) & set(jd_skills))
        missing = sorted(set(jd_skills) - set(resume_skills))
        resume_only = sorted(set(resume_skills) - set(jd_skills))

        skill_overlap_pct = (
            100.0 * len(matched) / len(jd_skills) if jd_skills else 0.0
        )
        sem_sim = self.semantic_similarity(resume_text, jd_text)

        overall = (
            SEMANTIC_WEIGHT * sem_sim + SKILL_OVERLAP_WEIGHT * skill_overlap_pct
        )

        return MatchResult(
            overall_score=round(overall, 1),
            semantic_similarity=round(sem_sim, 1),
            skill_overlap_pct=round(skill_overlap_pct, 1),
            matched_skills=matched,
            missing_skills=missing,
            resume_only_skills=resume_only,
        )
