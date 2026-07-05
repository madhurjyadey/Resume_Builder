"""
skills_data.py
--------------
A curated skills taxonomy used for keyword/fuzzy extraction from
resumes and job descriptions. Extend this list freely -- the matcher
treats it as the ground-truth vocabulary of "extractable" skills.

For a production version, swap this out for a proper taxonomy such as
ESCO (https://esco.ec.europa.eu/) or O*NET, or mine it dynamically
from a large corpus of job postings.
"""

TECH_SKILLS = [
    # Programming languages
    "python", "java", "c++", "c#", "javascript", "typescript", "r", "sql",
    "go", "rust", "scala", "matlab", "bash", "shell scripting",

    # ML / DL
    "machine learning", "deep learning", "neural networks", "nlp",
    "natural language processing", "computer vision", "reinforcement learning",
    "pytorch", "tensorflow", "keras", "scikit-learn", "xgboost", "lightgbm",
    "huggingface", "transformers", "bert", "sentence-bert", "gnn",
    "graph neural networks", "opencv", "spacy", "nltk",

    # Data
    "pandas", "numpy", "data analysis", "data visualization", "data cleaning",
    "etl", "data engineering", "big data", "spark", "hadoop", "airflow",
    "power bi", "tableau", "excel",

    # Web / backend
    "django", "flask", "fastapi", "react", "node.js", "rest api",
    "graphql", "html", "css",

    # Cloud / DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "ci/cd", "git", "github",
    "linux", "terraform", "jenkins",

    # Databases
    "mysql", "postgresql", "mongodb", "redis", "elasticsearch", "sqlite",

    # Other
    "agile", "scrum", "project management", "a/b testing", "statistics",
    "algorithm design", "optimization", "time series analysis",
]

SOFT_SKILLS = [
    "communication", "leadership", "teamwork", "problem solving",
    "critical thinking", "time management", "collaboration",
    "adaptability", "mentoring", "presentation skills", "stakeholder management",
]

ALL_SKILLS = sorted(set(TECH_SKILLS + SOFT_SKILLS))
