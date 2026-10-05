import importlib
import json
import re
from pathlib import Path
from typing import Any

import httpx
from langchain_core.exceptions import OutputParserException
from ollama import ResponseError
from pydantic import ValidationError

from models import ApplicantState


CONFIG_PATH = Path(__file__).with_name("workflow_config.json")
STEP_NAME_PATTERN = re.compile(r"step_[0-9]+_[a-z0-9_]+\Z")


def load_config(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    try:
        with config_path.open(encoding="utf-8") as config_file:
            config = json.load(config_file)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot load workflow config at {config_path}: {error}") from error

    if not isinstance(config, dict):
        raise ValueError("Workflow config must be a JSON object.")
    steps = config.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("'steps' must be a non-empty list in workflow_config.json.")
    if any(not isinstance(name, str) or not STEP_NAME_PATTERN.fullmatch(name) for name in steps):
        raise ValueError(
            "Each step must be named like 'step_01_collect_cv' and live in the steps folder."
        )
    if len(set(steps)) != len(steps):
        raise ValueError("Workflow step names must be unique.")
    return config


def run_workflow(
    config_path: Path = CONFIG_PATH,
    output_fn: Any = print,
    input_fn: Any = input,
) -> list[ApplicantState]:
    config = load_config(config_path)
    cv_folder = config.get("cv_folder")
    if not isinstance(cv_folder, str) or not cv_folder.strip():
        raise ValueError("'cv_folder' must be a non-empty folder path in workflow_config.json.")
    folder_path = Path(cv_folder)
    if not folder_path.is_absolute():
        folder_path = config_path.parent / folder_path
    if not folder_path.is_dir():
        raise ValueError(f"CV folder does not exist: {folder_path}")

    modules = []
    for step_name in config["steps"]:
        module = importlib.import_module(f"steps.{step_name}")
        run_step = getattr(module, "run", None)
        if not callable(run_step):
            raise RuntimeError(f"Workflow step '{step_name}' must define run(state, config).")
        modules.append((step_name, run_step))

    review_step_name = config.get("review_step")
    if not isinstance(review_step_name, str) or not STEP_NAME_PATTERN.fullmatch(
        review_step_name
    ):
        raise ValueError(
            "'review_step' must name a step module in workflow_config.json."
        )
    review_module = importlib.import_module(f"steps.{review_step_name}")
    review_applicants = getattr(review_module, "review_applicants", None)
    if not callable(review_applicants):
        raise RuntimeError(
            f"Review step '{review_step_name}' must define review_applicants(states, input_fn, output_fn)."
        )

    cv_paths = sorted(
        (path for path in folder_path.iterdir() if path.is_file() and path.suffix.casefold() == ".txt"),
        key=lambda path: path.name.casefold(),
    )
    output_fn("\nApplicant screening batch")
    output_fn(f"Reading .txt CVs from: {folder_path}")
    output_fn("Model skill extraction and rule results are advisory, not hiring decisions.")
    if not cv_paths:
        output_fn("No .txt CV files found; add CV files to the configured folder.")
        return []

    completed: list[ApplicantState] = []
    failed = 0
    recoverable_errors = (
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
    for cv_path in cv_paths:
        state = ApplicantState(cv_path=cv_path, output_fn=output_fn)
        try:
            for _, run_step in modules:
                state = run_step(state, config)
        except recoverable_errors as error:
            failed += 1
            output_fn(f"Could not process {cv_path.name}: {error}")
            continue
        completed.append(state)

    output_fn(
        f"Batch complete: {len(completed)} processed, {failed} failed. "
        "Now review each suggestion bin; the human reviewer makes the final decisions."
    )
    review_applicants(completed, input_fn, output_fn)
    return completed
