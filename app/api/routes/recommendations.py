"""
recommendations.py — Course recommendations using BERT semantic similarity.
POST /api/ml/recommend
"""
from __future__ import annotations

import time
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity

from app.api.middleware.auth import verify_api_key
from app.models.model_store import get_similarity_model

router = APIRouter()

COURSE_CATALOG = [
    "B.Tech Computer Science",
    "B.Tech Information Technology",
    "M.Tech AI & Machine Learning",
    "MBA Business Analytics",
    "Full Stack Web Development",
    "Cloud Computing & DevOps",
    "Cybersecurity Fundamentals",
    "React & Next.js Advanced",
    "Python for Data Science",
    "Blockchain & Web3 Development",
    "UI/UX Design Principles",
    "Project Management Professional",
    "System Design & Architecture",
    "Database Engineering",
    "Mobile App Development",
    "Data Engineering with Spark",
    "MLOps & Model Deployment",
    "DevSecOps Practices",
    "Microservices Architecture",
    "API Design & Development",
]


class RecommendRequest(BaseModel):
    student_id: str
    completed_courses: Optional[List[str]] = []


@router.post("/recommend")
async def get_recommendations(
    req: RecommendRequest,
    _: str = Depends(verify_api_key),
):
    t0 = time.time()
    completed = [c.strip() for c in (req.completed_courses or []) if c.strip()]

    # Candidates = catalog minus completed courses
    candidates = [c for c in COURSE_CATALOG if c not in completed]

    if not candidates:
        return {
            "student_id": req.student_id,
            "recommendations": [],
            "method": "sentence-transformers semantic similarity",
            "latency_ms": round((time.time() - t0) * 1000, 2),
        }

    if not completed:
        # No history → return top 5 from catalog
        recommendations = candidates[:5]
    else:
        model = get_similarity_model()

        # Encode completed courses → average embedding
        completed_embs = model.encode(completed)  # (n, 384)
        student_emb = np.mean(completed_embs, axis=0, keepdims=True)  # (1, 384)

        # Encode candidates
        candidate_embs = model.encode(candidates)  # (m, 384)

        # Cosine similarity between student profile and each candidate
        sims = cosine_similarity(student_emb, candidate_embs)[0]  # (m,)

        # Sort by descending similarity, take top 5
        top_indices = np.argsort(sims)[::-1][:5]
        recommendations = [candidates[i] for i in top_indices]

    return {
        "student_id": req.student_id,
        "recommendations": recommendations,
        "method": "sentence-transformers semantic similarity",
        "latency_ms": round((time.time() - t0) * 1000, 2),
    }
