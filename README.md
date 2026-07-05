# 🧩 Resume ↔ Job Description Matcher

An NLP tool that scores how well a resume matches a job description, using
Sentence-BERT embeddings for semantic similarity plus keyword-based skill
extraction to highlight matched and missing skills.

## Features
- Upload a resume as **PDF, DOCX, or TXT**
- Paste any job description
- Get:
  - An overall **match score** (0-100%)
  - A **semantic similarity** score (how close the overall content is)
  - A **skill overlap** score (% of JD skills found in the resume)
  - Lists of **matched skills**, **missing skills**, and skills you have that the JD doesn't mention
- Simple, interactive **Streamlit** UI with a gauge chart

## Architecture

```
resume_job_matcher/
├── app.py                 # Streamlit UI
├── requirements.txt
├── src/
│   ├── parser.py           # PDF/DOCX/TXT text extraction + cleaning
│   ├── skills_data.py       # Curated skills taxonomy (tech + soft skills)
│   └── matcher.py          # Embedding + skill-matching scoring engine
└── README.md
```

**Pipeline:**
1. `parser.py` extracts and cleans raw text from the uploaded resume and the pasted JD.
2. `matcher.py`:
   - Extracts skills from both texts against a curated taxonomy (`skills_data.py`), using word-boundary matching for short tokens (e.g. "R", "Go") and fuzzy matching for longer phrases.
   - Embeds both full texts with a Sentence-BERT model (`all-MiniLM-L6-v2`) and computes cosine similarity.
   - Combines semantic similarity + skill overlap into a final weighted score.
3. `app.py` renders the results interactively.

## Setup

```bash
# 1. Clone / unzip the project, then cd into it
cd resume_job_matcher

# 2. (Recommended) create a virtual environment
python3 -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The first run will download the Sentence-BERT model (~80MB) — this only happens once.

## Customizing

- **Skills taxonomy**: Edit `src/skills_data.py` to add/remove skills relevant to your target roles (e.g. add domain-specific tools).
- **Scoring weights**: Adjust `SEMANTIC_WEIGHT` and `SKILL_OVERLAP_WEIGHT` in `src/matcher.py` to change how much each signal contributes to the final score.
- **Embedding model**: Swap `all-MiniLM-L6-v2` for a larger model (e.g. `all-mpnet-base-v2`) in `ResumeJobMatcher.__init__` for higher accuracy at the cost of speed.

## Possible Extensions
- Fine-tune a classifier on labeled resume-JD pairs instead of a fixed weighted formula
- Add resume bullet-point rewrite suggestions using an LLM
- Extract skills via a proper NER model instead of a fixed taxonomy
- Support batch scoring (one JD vs. many resumes) for recruiter-side use cases
- Deploy to Streamlit Cloud or HuggingFace Spaces for a live public demo

## Tech Stack
Python · Sentence-Transformers · scikit-learn · RapidFuzz · pdfplumber · python-docx · Streamlit · Plotly

## License
MIT — free to use and modify for your own portfolio.
