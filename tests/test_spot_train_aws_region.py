"""Pins spot_train.sh to export AWS_REGION into every remote SSM step.

The .env-deprecation arc (#164 / PR 9d) deleted the sourced `.env`, so
AWS_REGION/AWS_DEFAULT_REGION — which boto3 and
`training/preflight.py::TrainingPreflight` (`check_env_vars("AWS_REGION")`)
hard-require — are no longer set in the spot shell unless each `run_ssm`
step's export line sets them explicitly. This was the #247 regression in
a sibling repo: 2026-05-16 Saturday SF PredictorTraining fast-failed at
`RuntimeError: Pre-flight: required env vars missing: ['AWS_REGION']`.

This test catches a future edit that drops the export from any step's
env line.
"""

from __future__ import annotations

import re
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent / "infrastructure" / "spot_train.sh"


def test_spot_train_exists():
    assert _SCRIPT.is_file()


def test_every_remote_export_line_sets_aws_region():
    text = _SCRIPT.read_text()
    # The per-step env line in each run_ssm heredoc.
    export_lines = [
        ln
        for ln in text.splitlines()
        if ln.startswith("export HOME=/home/ec2-user XDG_CACHE_HOME=/tmp")
    ]
    assert export_lines, "no per-step export line found — spot_train.sh structure changed"
    for ln in export_lines:
        assert "AWS_REGION=" in ln, (
            f"remote step export line missing AWS_REGION — boto3/TrainingPreflight "
            f"will fail on the SSM-launched spot (no .env post-#164). Line: {ln!r}"
        )
        assert "AWS_DEFAULT_REGION=" in ln, (
            f"remote step export line missing AWS_DEFAULT_REGION. Line: {ln!r}"
        )
