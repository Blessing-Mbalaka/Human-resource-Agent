import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from models import ApplicantState, SkillExtraction, SkillFinding
from steps.step_02_match_skills import run as match_skills
from steps.step_03_evaluate_rules import run as evaluate_rules
from workflow import run_workflow


class SkillMatchingTests(unittest.TestCase):
    def test_uses_model_skill_inference_and_keeps_verbatim_evidence(self) -> None:
        state = ApplicantState(
            cv_path=Path("candidate.txt"),
            cv_text="Built APIs using Python and FastAPI.",
        )
        extractor = MagicMock()
        extractor.invoke.return_value = SkillExtraction(
            findings=[
                SkillFinding(skill="Python", found=True, evidence="using Python"),
                SkillFinding(skill="SQL", found=False, evidence=""),
                SkillFinding(skill="FastAPI", found=True, evidence="FastAPI"),
            ]
        )

        with patch("steps.step_02_match_skills.ChatOllama") as chat_model:
            chat_model.return_value.with_structured_output.return_value = extractor
            match_skills(
                state,
                {
                    "model": "test-model",
                    "required_skills": ["Python", "SQL"],
                    "preferred_skills": ["FastAPI"],
                },
            )

        self.assertEqual(state.matched_required_skills, ["Python"])
        self.assertEqual(state.missing_required_skills, ["SQL"])
        self.assertEqual(state.matched_preferred_skills, ["FastAPI"])
        self.assertIn("Python", state.evidence)
        self.assertNotIn("SQL", state.evidence)

    def test_rejects_model_evidence_not_present_in_cv(self) -> None:
        state = ApplicantState(cv_path=Path("candidate.txt"), cv_text="Used PostgreSQL.")
        extractor = MagicMock()
        extractor.invoke.return_value = SkillExtraction(
            findings=[SkillFinding(skill="SQL", found=True, evidence="expert in SQL")]
        )

        with patch("steps.step_02_match_skills.ChatOllama") as chat_model:
            chat_model.return_value.with_structured_output.return_value = extractor
            with self.assertRaisesRegex(ValueError, "verbatim CV evidence"):
                match_skills(
                    state,
                    {
                        "model": "test-model",
                        "required_skills": ["SQL"],
                        "preferred_skills": [],
                    },
                )

    def test_fails_if_model_omits_a_configured_skill(self) -> None:
        state = ApplicantState(cv_path=Path("candidate.txt"), cv_text="Python developer.")
        extractor = MagicMock()
        extractor.invoke.return_value = SkillExtraction(
            findings=[SkillFinding(skill="Python", found=True, evidence="Python")]
        )

        with patch("steps.step_02_match_skills.ChatOllama") as chat_model:
            chat_model.return_value.with_structured_output.return_value = extractor
            with self.assertRaisesRegex(ValueError, "omitted configured skills"):
                match_skills(
                    state,
                    {
                        "model": "test-model",
                        "required_skills": ["Python", "SQL"],
                        "preferred_skills": [],
                    },
                )


class RuleEvaluationTests(unittest.TestCase):
    def test_first_matching_rule_sets_advisory_recommendation(self) -> None:
        state = ApplicantState(
            cv_path=Path("candidate.txt"),
            missing_required_skills=[],
            matched_required_skills=["Python", "SQL"],
            matched_preferred_skills=["FastAPI"],
        )
        messages: list[str] = []
        state.output_fn = messages.append

        evaluate_rules(
            state,
            {
                "routing_rules": [
                    {
                        "all": [
                            {"field": "missing_required_count", "operator": "equals", "value": 0},
                            {"field": "preferred_match_count", "operator": "gte", "value": 1},
                        ],
                        "recommendation": "Ready for human review",
                        "bin": "suggested_approval",
                    },
                    {
                        "all": [],
                        "recommendation": "Needs human review",
                        "bin": "suggested_rejection",
                    },
                ]
            },
        )

        self.assertEqual(state.recommendation, "Ready for human review")
        self.assertTrue(any("suggestion" in message.lower() for message in messages))


class WorkflowTests(unittest.TestCase):
    def test_workflow_batches_into_bins_then_records_human_decisions(self) -> None:
        output: list[str] = []
        reviewer_input = iter(
            [
                "approve",
                "Required skills verified against CV.",
                "reject",
                "Missing required SQL skill.",
            ]
        )

        def reviewer_input_fn(prompt: str) -> str:
            del prompt
            return next(reviewer_input)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            applicants = root / "applicants"
            applicants.mkdir()
            (applicants / "candidate.txt").write_text(
                "Python developer with SQL experience.", encoding="utf-8"
            )
            (applicants / "candidate_without_sql.txt").write_text(
                "Python developer.", encoding="utf-8"
            )
            config = {
                "steps": [
                    "step_01_collect_cv",
                    "step_02_match_skills",
                    "step_03_evaluate_rules",
                    "step_04_report",
                ],
                "review_step": "step_05_human_review",
                "cv_folder": "applicants",
                "model": "test-model",
                "required_skills": ["Python", "SQL"],
                "preferred_skills": ["FastAPI"],
                "routing_rules": [
                    {
                        "all": [
                            {
                                "field": "missing_required_count",
                                "operator": "equals",
                                "value": 0,
                            }
                        ],
                        "recommendation": "Meets required skills; review evidence",
                        "bin": "suggested_approval",
                    },
                    {
                        "all": [],
                        "recommendation": "Missing required skill evidence",
                        "bin": "suggested_rejection",
                    },
                ],
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            extractor = MagicMock()
            extractor.invoke.side_effect = [
                SkillExtraction(
                    findings=[
                        SkillFinding(skill="Python", found=True, evidence="Python"),
                        SkillFinding(skill="SQL", found=True, evidence="SQL"),
                        SkillFinding(skill="FastAPI", found=False, evidence=""),
                    ]
                ),
                SkillExtraction(
                    findings=[
                        SkillFinding(skill="Python", found=True, evidence="Python"),
                        SkillFinding(skill="SQL", found=False, evidence=""),
                        SkillFinding(skill="FastAPI", found=False, evidence=""),
                    ]
                ),
            ]
            with patch("steps.step_02_match_skills.ChatOllama") as chat_model:
                chat_model.return_value.with_structured_output.return_value = extractor
                states = run_workflow(
                    config_path,
                    output.append,
                    reviewer_input_fn,
                )

        self.assertEqual(len(states), 2)
        self.assertEqual(states[0].suggestion_bin, "suggested_approval")
        self.assertEqual(states[0].human_decision, "approve")
        self.assertEqual(states[1].suggestion_bin, "suggested_rejection")
        self.assertEqual(states[1].human_decision, "reject")
        self.assertEqual(states[1].human_reason, "Missing required SQL skill.")
        self.assertTrue(any("Suggested approval bin (1 applicant(s))" in message for message in output))
        self.assertTrue(any("Suggested rejection bin (1 applicant(s))" in message for message in output))
        self.assertTrue(any("2 processed, 0 failed" in message for message in output))


if __name__ == "__main__":
    unittest.main()
