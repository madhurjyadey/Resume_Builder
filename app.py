"""
app.py
------
Streamlit front-end for the Resume <-> Job Description Matcher.

Run locally with:
    streamlit run app.py
"""

import io
import streamlit as st
import plotly.graph_objects as go

from src.parser import extract_text, clean_text
from src.matcher import ResumeJobMatcher

st.set_page_config(page_title="Resume-JD Matcher", page_icon="🧩", layout="wide")


@st.cache_resource(show_spinner="Loading embedding model (first run only)...")
def load_matcher():
    return ResumeJobMatcher()


def gauge_chart(score: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#4C6EF5"},
                "steps": [
                    {"range": [0, 40], "color": "#FFE3E3"},
                    {"range": [40, 70], "color": "#FFF3BF"},
                    {"range": [70, 100], "color": "#D3F9D8"},
                ],
            },
            title={"text": "Overall Match Score"},
        )
    )
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def main():
    st.title("🧩 Resume ↔ Job Description Matcher")
    st.caption(
        "Upload a resume and paste a job description to see how well they align, "
        "which skills matched, and what's missing."
    )

    matcher = load_matcher()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📄 Resume")
        resume_file = st.file_uploader(
            "Upload resume (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"]
        )

    with col2:
        st.subheader("💼 Job Description")
        jd_text_input = st.text_area(
            "Paste the job description here", height=280,
            placeholder="Paste the full job posting text..."
        )

    if st.button("🔍 Analyze Match", type="primary", use_container_width=False):
        if not resume_file or not jd_text_input.strip():
            st.warning("Please upload a resume AND paste a job description.")
            return

        with st.spinner("Extracting text and scoring..."):
            file_bytes = io.BytesIO(resume_file.read())
            raw_resume_text = extract_text(resume_file.name, file_bytes=file_bytes)
            resume_text = clean_text(raw_resume_text)
            jd_text = clean_text(jd_text_input)

            result = matcher.score(resume_text, jd_text)

        st.divider()

        score_col, chart_col = st.columns([1, 1])
        with score_col:
            st.plotly_chart(gauge_chart(result.overall_score), use_container_width=True)
            m1, m2 = st.columns(2)
            m1.metric("Semantic Similarity", f"{result.semantic_similarity}%")
            m2.metric("Skill Overlap", f"{result.skill_overlap_pct}%")

        with chart_col:
            st.subheader("✅ Matched Skills")
            if result.matched_skills:
                st.write(", ".join(f"`{s}`" for s in result.matched_skills))
            else:
                st.write("No direct skill matches found.")

            st.subheader("❌ Missing Skills (in JD, not in Resume)")
            if result.missing_skills:
                st.write(", ".join(f"`{s}`" for s in result.missing_skills))
            else:
                st.write("None — great coverage!")

            with st.expander("Skills in your resume not mentioned in this JD"):
                if result.resume_only_skills:
                    st.write(", ".join(f"`{s}`" for s in result.resume_only_skills))
                else:
                    st.write("None.")

        st.divider()
        st.info(
            "💡 **Tip:** Try weaving the missing skills above into your resume "
            "(only if you genuinely have that experience!) to improve your match score "
            "for this specific role."
        )


if __name__ == "__main__":
    main()
