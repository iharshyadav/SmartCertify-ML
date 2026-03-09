"""
SmartCertify ML — Recommendation Engine
Content-based and collaborative filtering for course/certificate recommendations.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from app.config.settings import RANDOM_SEED, RECOMMENDATION_DATASET_PATH, MODEL_DIR
from app.utils.model_io import save_sklearn_model, load_sklearn_model

logger = logging.getLogger(__name__)

_tfidf = None
_course_matrix = None
_interaction_data = None


def _load_interaction_data() -> Optional[pd.DataFrame]:
    """Load student interaction data."""
    global _interaction_data
    if _interaction_data is None:
        try:
            _interaction_data = pd.read_csv(RECOMMENDATION_DATASET_PATH)
        except FileNotFoundError:
            logger.warning("Recommendation dataset not found")
            return None
    return _interaction_data


def train_recommender(df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Train the recommendation models (content-based + collaborative)."""
    if df is None:
        df = _load_interaction_data()
        if df is None:
            return {"error": "No interaction data available"}

    # Content-based: TF-IDF on course names + skills
    courses = df.drop_duplicates("course_name")[["course_name", "course_id", "skills_gained"]].copy()
    courses["text"] = courses["course_name"] + " " + courses["skills_gained"].fillna("")

    tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2), stop_words="english")
    course_vectors = tfidf.fit_transform(courses["text"])

    # Compute course similarity matrix
    course_sim = cosine_similarity(course_vectors)

    save_sklearn_model({
        "tfidf": tfidf,
        "course_vectors": course_vectors,
        "course_sim": course_sim,
        "courses": courses.reset_index(drop=True),
    }, "recommender.joblib")

    # Collaborative filtering: SVD matrix factorization
    # Build user-course rating matrix
    pivot = df.pivot_table(index="student_id", columns="course_id", values="rating", aggfunc="mean")
    pivot = pivot.fillna(0)

    # SVD
    n_factors = min(20, min(pivot.shape) - 1)
    if n_factors > 0:
        U, sigma, Vt = svds(pivot.values.astype(float), k=n_factors)
        sigma_diag = np.diag(sigma)
        predicted_ratings = U @ sigma_diag @ Vt

        save_sklearn_model({
            "predicted_ratings": predicted_ratings,
            "student_ids": list(pivot.index),
            "course_ids": list(pivot.columns),
        }, "collaborative_filter.joblib")

    metrics = {
        "n_courses": len(courses),
        "n_students": df["student_id"].nunique(),
        "n_interactions": len(df),
        "n_svd_factors": n_factors,
    }

    logger.info(f"Recommender trained: {metrics}")
    return metrics


def recommend_content_based(
    completed_courses: List[str],
    n_recommendations: int = 5,
) -> List[Dict[str, Any]]:
    """Content-based recommendations using TF-IDF similarity."""
    data = load_sklearn_model("recommender.joblib")
    if data is None:
        return []

    courses = data["courses"]
    course_sim = data["course_sim"]

    # Find indices of completed courses
    completed_indices = []
    for course in completed_courses:
        matches = courses[courses["course_name"].str.lower() == course.lower()]
        if len(matches) > 0:
            completed_indices.append(matches.index[0])

    if not completed_indices:
        # Return most popular courses as fallback
        return [
            {"name": row["course_name"], "score": 0.5, "reason": "Popular course"}
            for _, row in courses.head(n_recommendations).iterrows()
        ]

    # Aggregate similarity scores
    sim_scores = np.zeros(len(courses))
    for idx in completed_indices:
        if idx < len(course_sim):
            sim_scores += course_sim[idx]

    # Exclude completed courses
    for idx in completed_indices:
        sim_scores[idx] = -1

    # Get top-N
    top_indices = np.argsort(sim_scores)[-n_recommendations:][::-1]

    recommendations = []
    for idx in top_indices:
        if sim_scores[idx] > 0 and idx < len(courses):
            row = courses.iloc[idx]
            recommendations.append({
                "name": row["course_name"],
                "score": round(float(sim_scores[idx] / len(completed_indices)), 4),
                "reason": f"Similar to your completed courses",
            })

    return recommendations


def recommend_collaborative(
    student_id: str,
    n_recommendations: int = 5,
) -> List[Dict[str, Any]]:
    """Collaborative filtering recommendations using SVD."""
    cf_data = load_sklearn_model("collaborative_filter.joblib")
    if cf_data is None:
        return []

    student_ids = cf_data["student_ids"]
    course_ids = cf_data["course_ids"]
    predicted_ratings = cf_data["predicted_ratings"]

    if student_id not in student_ids:
        return []

    student_idx = student_ids.index(student_id)
    ratings = predicted_ratings[student_idx]

    # Get interaction data to exclude already completed
    df = _load_interaction_data()
    if df is not None:
        completed = set(df[df["student_id"] == student_id]["course_id"].values)
    else:
        completed = set()

    # Rank courses
    course_scores = []
    for i, (course_id, score) in enumerate(zip(course_ids, ratings)):
        if course_id not in completed:
            course_scores.append((course_id, float(score)))

    course_scores.sort(key=lambda x: x[1], reverse=True)

    # Get course names
    recommender_data = load_sklearn_model("recommender.joblib")
    courses = recommender_data["courses"] if recommender_data else pd.DataFrame()

    recommendations = []
    for course_id, score in course_scores[:n_recommendations]:
        name_matches = courses[courses["course_id"] == course_id]
        name = name_matches["course_name"].iloc[0] if len(name_matches) > 0 else f"Course {course_id}"
        recommendations.append({
            "name": name,
            "score": round(score, 4),
            "reason": "Recommended based on similar students",
        })

    return recommendations


def get_recommendations(
    student_id: str,
    completed_courses: List[str],
    n_recommendations: int = 5,
) -> Dict[str, Any]:
    """Get combined recommendations from both methods."""
    content_recs = recommend_content_based(completed_courses, n_recommendations)
    collab_recs = recommend_collaborative(student_id, n_recommendations)

    # Merge and deduplicate
    seen = set()
    combined = []
    for rec in content_recs + collab_recs:
        if rec["name"] not in seen:
            seen.add(rec["name"])
            combined.append(rec)

    # Sort by score and limit
    combined.sort(key=lambda x: x["score"], reverse=True)
    combined = combined[:n_recommendations]

    return {"recommendations": combined}


def main():
    """Train recommendation engine."""
    if not Path(RECOMMENDATION_DATASET_PATH).exists():
        from app.data.generate_synthetic import generate_recommendation_dataset
        df = generate_recommendation_dataset()
        df.to_csv(RECOMMENDATION_DATASET_PATH, index=False)

    print("Training recommendation engine...")
    results = train_recommender()
    print(f"\n✅ Recommendation engine trained!")
    for k, v in results.items():
        print(f"   {k}: {v}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
