from typing import Any

from models import ApplicantState


def _matches(actual: Any, operator: str, expected: Any) -> bool:
    if operator == "equals":
        return actual == expected
    if operator == "not_equals":
        return actual != expected
    if operator in {"gt", "gte", "lt", "lte"}:
        if not isinstance(actual, (int, float)) or not isinstance(expected, (int, float)):
            raise ValueError(f"Operator '{operator}' requires numeric values.")
        return {
            "gt": actual > expected,
            "gte": actual >= expected,
            "lt": actual < expected,
            "lte": actual <= expected,
        }[operator]
    if operator == "contains":
        if not isinstance(actual, (str, list, tuple, set)):
            raise ValueError("'contains' requires a string or collection field.")
        return expected in actual
    raise ValueError(f"Unsupported routing operator: {operator}")


def run(state: ApplicantState, config: dict[str, Any]) -> ApplicantState:
    metrics: dict[str, int | list[str]] = {
        "missing_required_count": len(state.missing_required_skills),
        "required_match_count": len(state.matched_required_skills),
        "preferred_match_count": len(state.matched_preferred_skills),
        "missing_required_skills": state.missing_required_skills,
    }
    rules = config.get("routing_rules")
    if not isinstance(rules, list) or not rules:
        raise ValueError("'routing_rules' must be a non-empty list.")

    for rule in rules:
        if not isinstance(rule, dict) or not isinstance(rule.get("all"), list):
            raise ValueError("Each routing rule must have an 'all' list of conditions.")
        conditions = rule["all"]
        matched = True
        for condition in conditions:
            if not isinstance(condition, dict):
                raise ValueError("Each routing condition must be a JSON object.")
            field = condition.get("field")
            operator = condition.get("operator")
            if field not in metrics:
                raise ValueError(f"Unknown routing field: {field}")
            if not isinstance(operator, str) or "value" not in condition:
                raise ValueError("Each condition needs an operator and a value.")
            matched = matched and _matches(metrics[field], operator, condition["value"])
            if not matched:
                break
        if matched:
            recommendation = rule.get("recommendation")
            if not isinstance(recommendation, str) or not recommendation.strip():
                raise ValueError("A matched routing rule needs a recommendation label.")
            suggestion_bin = rule.get("bin")
            if not isinstance(suggestion_bin, str) or suggestion_bin not in {
                "suggested_approval",
                "suggested_rejection",
            }:
                raise ValueError(
                    "A matched routing rule must set bin to suggested_approval "
                    "or suggested_rejection."
                )
            state.recommendation = recommendation
            state.suggestion_bin = suggestion_bin
            break
    else:
        raise ValueError("No routing rule matched; add a final fallback rule.")

    state.output_fn(
        f"Criteria suggestion ({state.suggestion_bin}): {state.recommendation}"
    )
    return state
