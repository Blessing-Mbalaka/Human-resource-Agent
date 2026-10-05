from collections.abc import Callable, Sequence

from models import ApplicantState


BIN_LABELS = {
    "suggested_approval": "Suggested approval bin",
    "suggested_rejection": "Suggested rejection bin",
}
DECISIONS = {"approve", "reject", "hold"}


def _show_applicant(state: ApplicantState, output_fn: Callable[[str], None]) -> None:
    output_fn(f"\nCV: {state.cv_path.name}")
    output_fn(f"Criteria suggestion: {state.recommendation}")
    output_fn(
        "Required skills found: "
        + (", ".join(state.matched_required_skills) or "None")
    )
    output_fn(
        "Required skills without evidence: "
        + (", ".join(state.missing_required_skills) or "None")
    )
    output_fn(
        "Preferred skills found: "
        + (", ".join(state.matched_preferred_skills) or "None")
    )
    for skill, evidence in state.evidence.items():
        output_fn(f"Evidence for {skill}: {evidence}")
    output_fn("Review the original CV before recording a final decision.")


def review_applicants(
    states: Sequence[ApplicantState],
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> None:
    for suggestion_bin, label in BIN_LABELS.items():
        applicants = [state for state in states if state.suggestion_bin == suggestion_bin]
        output_fn(f"\n{label} ({len(applicants)} applicant(s))")
        for state in applicants:
            _show_applicant(state, output_fn)
            while True:
                decision = input_fn(
                    "Human decision [approve/reject/hold]: "
                ).strip().casefold()
                if decision in DECISIONS:
                    break
                output_fn("Enter approve, reject, or hold.")

            reason = input_fn("Reason for the decision: ").strip()
            while decision == "reject" and not reason:
                output_fn("A job-related reason is required when rejecting.")
                reason = input_fn("Reason for rejection: ").strip()

            state.human_decision = decision
            state.human_reason = reason
            output_fn(f"Human decision recorded: {decision}")
