from typing import Any

from models import ApplicantState


def run(state: ApplicantState, config: dict[str, Any]) -> ApplicantState:
    del config
    state.cv_text = state.cv_path.read_text(encoding="utf-8").strip()
    if not state.cv_text:
        raise ValueError("CV file is empty.")
    return state
