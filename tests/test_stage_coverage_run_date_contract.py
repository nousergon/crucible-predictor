"""Released Krepis contract required by this stage-coverage consumer."""

from __future__ import annotations

import pytest

from krepis.stage_coverage import StageVerdict, verdict_key


def test_stage_verdict_requires_the_execution_run_date_for_its_exact_s3_key() -> None:
    """I8155: no caller may fall back from execution identity to cycle date."""
    with pytest.raises(ValueError, match="run_date"):
        StageVerdict(stage="ConsumerContract", status="COVERED", run_date="")

    verdict = StageVerdict(
        stage="ConsumerContract",
        status="COVERED",
        run_date="2026-08-29",
        cycle_date="2026-08-28",
    )
    assert verdict_key(verdict) == "_stage_coverage/2026-08-29/ConsumerContract.json"
