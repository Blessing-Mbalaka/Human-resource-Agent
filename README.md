# Applicant screening console prototype

This local console prototype uses LangChain with Ollama for small talk and
model-based extraction of job-related skill evidence. Explicit Python rules
evaluate that extracted evidence and print a recommendation for a human
reviewer. It does not make accept/reject employment decisions.

## Setup

Activate the project's `venv` and install dependencies:

```powershell
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Ensure Ollama is running and the configured local model is installed:

```powershell
ollama pull llama3.2:1b
python Tev_Agent.py
```

Place one UTF-8 plain-text CV per `.txt` file in `applicants/`, then run
`python Tev_Agent.py`. The app extracts configured skill evidence and evaluates
the criteria for the whole batch, then presents suggested-approval and
suggested-rejection bins. A human reviewer chooses approve, reject, or hold for
each CV and enters a reason. `reject` requires a reason. To use small talk and
trigger batches with `screen applicants`, run `python Tev_Agent.py --chat`.

## Editing the workflow

- Edit `required_skills` and `preferred_skills` in `workflow_config.json`.
- Edit `routing_rules` to change advisory routing. Rules are checked top to
  bottom; the first matching rule wins. Each rule's `all` list is an AND of
  conditions. An empty `all` list is a fallback. Each rule must set `bin` to
  `suggested_approval` or `suggested_rejection`.
- Supported routing fields: `missing_required_count`, `required_match_count`,
  `preferred_match_count`, and `missing_required_skills`.
- Supported operators: `equals`, `not_equals`, `gt`, `gte`, `lt`, `lte`, and
  `contains`.
- Set `model` to an installed Ollama model name.
- Change the order of modules in `steps` to change the workflow order. Each
  module in `steps/` must define `run(state, config)` and return the state.

The model extracts only configured skills and must return verbatim CV excerpts
as evidence. The workflow rejects unsupported evidence and displays the source
file and excerpts so a reviewer can verify them. Model extraction is not
verification that a candidate actually has a qualification. CV content is
processed in memory and is not written to disk by this prototype.

The suggested bins are driven by the configured, job-related rules—not by an
LLM's general impression of suitability. The model extracts evidence only;
Python checks that reported quotes appear in the CV before using them. Review
the original CV before deciding. Human decisions and reasons remain in memory
for this run and are not saved to a database or file.

This is a prototype, not a validated hiring system. Keep criteria job-related,
review the underlying CV evidence, and leave employment decisions to a human.

## Tests

```powershell
python -m unittest discover -s tests -v
```
