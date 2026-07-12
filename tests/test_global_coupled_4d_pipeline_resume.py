from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_unified_pipeline_has_distinct_resumable_smoke_stages():
    unified = (ROOT / "scripts/run_global_coupled_4d_smoke_and_matched.sh").read_text(
        encoding="utf-8"
    )
    smoke = (ROOT / "scripts/run_global_coupled_4d_smoke.sh").read_text(
        encoding="utf-8"
    )
    formal = (ROOT / "scripts/run_global_coupled_4d_formal_matched.sh").read_text(
        encoding="utf-8"
    )

    for marker in (
        "SMOKE_TRAIN_COMPLETED",
        "SMOKE_SAMPLE_COMPLETED",
        "SMOKE_EVAL_COMPLETED",
    ):
        assert marker in unified
        assert marker in smoke
    assert "checkpoints/last.ckpt" in smoke
    assert "repair_sample_payload_provenance.py" in smoke
    assert smoke.index("repair_sample_payload_provenance.py") < smoke.index(
        "sample_global_coupled_4d_flow.py"
    )
    assert "SMOKE_EVAL_COMPLETED" in formal
    assert "run_global_coupled_4d_formal_matched.sh" in unified


def test_stale_failure_is_archived_before_new_run():
    unified = (ROOT / "scripts/run_global_coupled_4d_smoke_and_matched.sh").read_text(
        encoding="utf-8"
    )
    assert "failure_history" in unified
    assert "shutil.move" in unified
    assert unified.index("shutil.move") < unified.index("trap fail ERR")
