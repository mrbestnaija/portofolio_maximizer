# Phase 9 Pipeline Hardening

Date: 2026-03-24

## Scope

This patch hardens the classifier bootstrap path before ETL, labeling, training,
and evaluation proceed. The main focus is fail-closed behavior for filename
ambiguity, ticker-scoped timestamp/date validation, and unreadable dataset
handling.

## What Changed

- `scripts/validate_pipeline_inputs.py`
  - V3 alignment is now ticker-scoped when ticker metadata is available.
  - A single foreign ticker parquet no longer satisfies another ticker.
  - Empty or unreadable parquet coverage now returns a clean `FAIL` instead of
    crashing.
  - `signal_id` and `ts_signal_id` can supply ticker context when `ticker` is
    missing in JSONL.
  - V6 now warns on malformed timestamps separately from null timestamps.
- `scripts/generate_classifier_training_labels.py`
  - Auto-parquet selection now distinguishes ticker-named files from unnamed
    generic files.
  - Multi-ticker runs refuse generic fallback and require ticker-named files or
    an explicit `--parquet`.
  - Ambiguous generic checkpoint sets fail closed instead of picking the first
    parquet.
  - Summary JSON now records requested tickers, per-ticker outcomes, and whether
    `--allow-partial` was used.
- `scripts/train_directional_classifier.py`
  - Corrupt or unreadable parquet datasets return `dataset_unreadable`.
- `scripts/evaluate_directional_classifier.py`
  - Corrupt or unreadable parquet datasets return `dataset_unreadable`.
- `bash/overnight_classifier_bootstrap.ps1`
  - Validator exit code `2` is now a blocking error instead of a warning.

## Concurrent Lane Check

The patch was prepared in a clean worktree based on `origin/master` to avoid
overlapping with the dirty parallel Claude/LLM lane. The concurrent edits found
in the other worktree were limited to `ai_llm/`, `config/openclaw_*`, and LLM
operator tooling, so this pipeline hardening patch stays isolated from that
ongoing work.

## Operator Impact

- If checkpoint filenames do not contain ticker identifiers, multi-ticker label
  generation now stops and asks for explicit disambiguation.
- If evaluation dates exceed actual parquet coverage, the validator reports that
  clearly per ticker instead of implicitly borrowing another ticker's data.
- If the dataset parquet is corrupt, training and evaluation return structured
  errors instead of bubbling raw Arrow exceptions.

## Verification

Focused regressions were verified with:

- `python -m pytest tests/scripts/test_validate_pipeline_inputs.py -q --basetemp C:\tmp\pmx_validate_hardening`
- `python -m pytest tests/scripts/test_generate_classifier_training_labels.py -q --basetemp C:\tmp\pmx_labels_hardening`
- `python -m pytest tests/scripts/test_train_directional_classifier.py -q --basetemp C:\tmp\pmx_train_hardening`
- `python -m pytest tests/scripts/test_evaluate_directional_classifier.py -q --basetemp C:\tmp\pmx_eval_hardening`
- `python -m py_compile scripts/validate_pipeline_inputs.py scripts/generate_classifier_training_labels.py scripts/train_directional_classifier.py scripts/evaluate_directional_classifier.py`
- `git diff --check`
