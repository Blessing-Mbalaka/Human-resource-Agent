import json
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from flask import Flask, abort, flash, redirect, render_template, request, session, url_for
from langchain_core.exceptions import OutputParserException
from ollama import ResponseError
from pydantic import ValidationError

import workflow
from models import ApplicantState
from steps import step_01_collect_cv, step_02_match_skills, step_03_evaluate_rules

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_PATH = DATA_DIR / "results.json"
NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
MAX_CV_CHARS = 20000
DECISIONS = {"approve", "reject", "hold"}
BINS = {
    "suggested_approval": "Suggested approval bin",
    "suggested_rejection": "Suggested rejection bin",
}
SCREEN_ERRORS = (
    OSError,
    ValueError,
    RuntimeError,
    ConnectionError,
    TimeoutError,
    httpx.HTTPError,
    ResponseError,
    OutputParserException,
    ValidationError,
)

app = Flask(__name__)
app.config["SECRET_KEY"] = secrets.token_hex(32)


def cv_folder() -> Path:
    config = workflow.load_config()
    folder = Path(str(config.get("cv_folder", "applicants")))
    if not folder.is_absolute():
        folder = workflow.CONFIG_PATH.parent / folder
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def cv_path(name: str) -> Path:
    if not NAME_PATTERN.fullmatch(name):
        abort(404)
    path = cv_folder() / f"{name}.txt"
    if not path.is_file():
        abort(404)
    return path


def load_results() -> dict[str, Any]:
    try:
        return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save_results(results: dict[str, Any]) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")


def csrf_token() -> str:
    if "csrf" not in session:
        session["csrf"] = secrets.token_hex(16)
    return session["csrf"]


app.jinja_env.globals["csrf_token"] = csrf_token
app.jinja_env.globals["BINS"] = BINS


@app.before_request
def check_csrf() -> None:
    if request.method == "POST":
        sent = request.form.get("csrf", "")
        if not sent or not secrets.compare_digest(sent, session.get("csrf", "")):
            abort(400)


def valid_cv_text(text: str) -> str | None:
    text = text.strip()
    if not text:
        flash("CV text cannot be empty.", "error")
        return None
    if len(text) > MAX_CV_CHARS:
        flash(f"CV text is limited to {MAX_CV_CHARS} characters.", "error")
        return None
    return text


@app.get("/")
def index() -> str:
    results = load_results()
    names = sorted((p.stem for p in cv_folder().glob("*.txt")), key=str.casefold)
    groups: dict[str, list[tuple[str, dict[str, Any]]]] = {key: [] for key in BINS}
    groups["unscreened"] = []
    for name in names:
        result = results.get(name)
        key = result["suggestion_bin"] if result else "unscreened"
        groups.setdefault(key, []).append((name, result or {}))
    return render_template("index.html", groups=groups)


@app.route("/applicants/new", methods=["GET", "POST"])
def create() -> Any:
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        text = valid_cv_text(request.form.get("cv_text", ""))
        if not NAME_PATTERN.fullmatch(name):
            flash("Name must be letters, digits, '-' or '_' (max 64).", "error")
        elif (cv_folder() / f"{name}.txt").exists():
            flash("An applicant with that name already exists.", "error")
        elif text:
            (cv_folder() / f"{name}.txt").write_text(text, encoding="utf-8")
            flash(f"Applicant {name} created.", "ok")
            return redirect(url_for("detail", name=name))
        return render_template(
            "form.html", title="New applicant", name=name,
            cv_text=request.form.get("cv_text", ""), editing=False,
        )
    return render_template("form.html", title="New applicant", name="", cv_text="", editing=False)


@app.get("/applicants/<name>")
def detail(name: str) -> str:
    path = cv_path(name)
    return render_template(
        "detail.html", name=name, cv_text=path.read_text(encoding="utf-8"),
        result=load_results().get(name),
    )


@app.route("/applicants/<name>/edit", methods=["GET", "POST"])
def edit(name: str) -> Any:
    path = cv_path(name)
    if request.method == "POST":
        text = valid_cv_text(request.form.get("cv_text", ""))
        if text:
            path.write_text(text, encoding="utf-8")
            results = load_results()
            if results.pop(name, None) is not None:
                save_results(results)
                flash("CV updated; previous screening result cleared. Screen again.", "ok")
            else:
                flash("CV updated.", "ok")
            return redirect(url_for("detail", name=name))
        return render_template(
            "form.html", title=f"Edit {name}", name=name,
            cv_text=request.form.get("cv_text", ""), editing=True,
        )
    return render_template(
        "form.html", title=f"Edit {name}", name=name,
        cv_text=path.read_text(encoding="utf-8"), editing=True,
    )


@app.post("/applicants/<name>/delete")
def delete(name: str) -> Any:
    cv_path(name).unlink()
    results = load_results()
    if results.pop(name, None) is not None:
        save_results(results)
    flash(f"Applicant {name} deleted.", "ok")
    return redirect(url_for("index"))


@app.post("/applicants/<name>/screen")
def screen(name: str) -> Any:
    path = cv_path(name)
    config = workflow.load_config()
    state = ApplicantState(cv_path=path, output_fn=lambda _message: None)
    try:
        for step in (step_01_collect_cv, step_02_match_skills, step_03_evaluate_rules):
            state = step.run(state, config)
    except SCREEN_ERRORS as error:
        flash(f"Could not screen {name}: {error}", "error")
        return redirect(url_for("detail", name=name))
    results = load_results()
    results[name] = {
        "suggestion_bin": state.suggestion_bin,
        "recommendation": state.recommendation,
        "matched_required": state.matched_required_skills,
        "matched_preferred": state.matched_preferred_skills,
        "missing_required": state.missing_required_skills,
        "evidence": state.evidence,
        "screened_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "human_decision": "",
        "human_reason": "",
    }
    save_results(results)
    flash("Screening complete. Suggestions are advisory; record your decision.", "ok")
    return redirect(url_for("detail", name=name))


@app.post("/applicants/<name>/decide")
def decide(name: str) -> Any:
    cv_path(name)
    results = load_results()
    result = results.get(name)
    decision = request.form.get("decision", "")
    reason = request.form.get("reason", "").strip()[:1000]
    if result is None:
        flash("Screen this applicant first.", "error")
    elif decision not in DECISIONS:
        flash("Choose approve, reject or hold.", "error")
    elif decision == "reject" and not reason:
        flash("A reason is required to reject.", "error")
    else:
        result["human_decision"] = decision
        result["human_reason"] = reason
        save_results(results)
        flash(f"Decision recorded: {decision}.", "ok")
    return redirect(url_for("detail", name=name))


if __name__ == "__main__":
    app.run(debug=False)
