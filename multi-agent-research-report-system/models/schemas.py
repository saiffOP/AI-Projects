from typing import List
from pydantic import BaseModel


class ResearchPlan(BaseModel):
    topic: str
    objective: str
    research_tasks: List[str]


class ResearchFinding(BaseModel):
    title: str
    summary: str
    source_url: str


class ResearchResult(BaseModel):
    topic: str
    findings: List[ResearchFinding]
    gaps: List[str]


class AnalysisResult(BaseModel):
    key_findings: List[str]
    comparisons: List[str]
    recommendation: str
    missing_information: List[str]


class ReviewResult(BaseModel):
    approved: bool
    score: int
    feedback: List[str]