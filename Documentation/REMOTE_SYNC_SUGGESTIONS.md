# Remote Sync Suggestions

The worktree currently lags behind the primary local deployment. To minimize merge conflicts
and duplicated effort, apply the following suggestions in the ground-truth repository before
pushing the next sync commit:

## Documentation & Onboarding
- Update `README.md` to reflect the actual project name (`portfolio_maximizer`), maturity
  status, and current test suite coverage rather than the legacy “v45” marketing copy.
- Double-check onboarding steps so that contributors can clone the repo and run the pipeline
  without guessing directory names or prerequisites.

## Pipeline Entry Point
- Move the logging setup in `scripts/run_etl_pipeline.py` back behind a `main()` guard or a
  dedicated initializer so importing the orchestrator from other modules or notebooks does not
  mutate global logging handlers.
- Extract reusable pipeline orchestration logic into a class or helper function that can be
  exercised directly from tests, leaving Click-specific argument parsing inside the CLI wrapper.

## Data Persistence & Auditing
- Adjust the naming scheme in `etl/data_storage.py` so that saved parquet files include a
  timestamp or unique run identifier, preventing silent overwrites during multiple runs on the
  same day.
- Persist run metadata (config hash, data source, execution mode) alongside artifacts to simplify
  troubleshooting and historical comparisons.

## LLM Integration Ergonomics
- Expand the Ollama prerequisite documentation and provide a graceful failure mode so enabling
  `--enable-llm` without the local server does not immediately abort the pipeline.
- Consider injecting the LLM clients via dependency inversion to make it easier to stub or mock
  them in automated tests.

Documenting these tasks in the ground-truth repository before performing the merge will keep both
local environments aligned and avoid re-solving the same issues after synchronization.
