from typing import Any

from models import ApplicantState


def run(state: ApplicantState, config: dict[str, Any]) -> ApplicantState:
    del config
    state.output_fn(f"\nApplicant file: {state.cv_path.name}")
    state.output_fn(
        "Required skills supported by CV: "
        + (", ".join(state.matched_required_skills) or "None")
    )
    state.output_fn(
        "Required skills without evidence: "
        + (", ".join(state.missing_required_skills) or "None")
    )
    state.output_fn(
        "Preferred skills supported by CV: "
        + (", ".join(state.matched_preferred_skills) or "None")
    )
    for skill, evidence in state.evidence.items():
        state.output_fn(f"Evidence for {skill}: {evidence}")
    state.output_fn(f"Recommendation for human review: {state.recommendation}")
    state.output_fn("This is not an employment decision.")
    return state
