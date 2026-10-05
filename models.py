from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, Field


class SkillFinding(BaseModel):
    skill: str = Field(description="One configured job-related skill.")
    found: bool = Field(description="Whether the CV supports this skill.")
    evidence: str = Field(
        description="An exact excerpt copied from the CV, or an empty string if not found."
    )


class SkillExtraction(BaseModel):
    findings: list[SkillFinding]


@dataclass
class ApplicantState:
    cv_path: Path
    cv_text: str = ""
    matched_required_skills: list[str] = field(default_factory=list)
    matched_preferred_skills: list[str] = field(default_factory=list)
    missing_required_skills: list[str] = field(default_factory=list)
    evidence: dict[str, str] = field(default_factory=dict)
    recommendation: str = ""
    suggestion_bin: str = ""
    human_decision: str = ""
    human_reason: str = ""
    output_fn: Callable[[str], None] = field(default=print, repr=False)
