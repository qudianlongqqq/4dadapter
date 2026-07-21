import json
import shutil
from pathlib import Path

import pytest
import torch

from scripts.verify_mcvr_v8_multiseed_run import verify_run


def _completed_run(tmp_path, *, seed=12):
    output = tmp_path / f"full_seed{seed}"
    checkpoints = output / "checkpoints"
    fast_dir = output / "validation_cache" / "step010000" / "fast"
    validation_dir = output / "validation_cache" / "step012500" / "full"
    checkpoints.mkdir(parents=True)
    fast_dir.mkdir(parents=True)
    validation_dir.mkdir(parents=True)
    status = {
        "status": "MCVR_V8_FULL_V1_FORMAL_LARGE_12P5K_COMPLETED",
        "phase": "COMPLETED",
        "actual_completed_step": 12500,
        "planned_original_total_steps": 200000,
        "effective_batch": 64,
        "total_record_exposure": 800000,
        "latest_validation": {"mode": "FULL"},
        "elapsed_seconds": 123.0,
        "formal_test_records_read": 0,
        "formal_test_assets_opened": False,
        "frozen_holdout_records_read": 0,
    }
    (output / "status.json").write_text(json.dumps(status), encoding="utf-8")
    checkpoint = {
        "step": 12500,
        "model_state_dict": {},
        "optimizer_state_dict": {},
        "scheduler_state_dict": None,
        "rng_states": {"python": (), "numpy": (), "torch": torch.get_rng_state()},
        "sampler_state": {
            "records_exposed": 800000,
            "next_record_exposure_offset": 800000,
        },
    }
    step_path = checkpoints / "step012500.ckpt"
    torch.save(checkpoint, step_path)
    shutil.copyfile(step_path, checkpoints / "last.ckpt")
    validation = {
        "mode": "FULL",
        "records": 10000,
        "formal_test_records_read": 0,
        "formal_test_assets_opened": False,
        "frozen_holdout_records_read": 0,
    }
    fast = {**validation, "mode": "FAST", "records": 1000}
    (fast_dir / "evaluation.json").write_text(json.dumps(fast), encoding="utf-8")
    (validation_dir / "evaluation.json").write_text(
        json.dumps(validation), encoding="utf-8"
    )
    return output


def test_completed_run_verifier_binds_checkpoint_validation_and_exit_code(tmp_path):
    output = _completed_run(tmp_path)
    result = verify_run(output, seed=12, exit_code=0)
    assert result["status"] == "COMPLETED"
    assert result["checkpoint_sha256"]
    assert result["validation_sha256"]
    assert result["runtime_seconds"] == 123.0
    assert result["exit_code"] == 0


def test_completed_run_verifier_fails_closed_on_isolation_violation(tmp_path):
    output = _completed_run(tmp_path)
    status_path = output / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["frozen_holdout_records_read"] = 1
    status_path.write_text(json.dumps(status), encoding="utf-8")
    with pytest.raises(RuntimeError, match="frozen_holdout_records_read"):
        verify_run(output, seed=12, exit_code=0)


def test_powershell_orchestrator_is_serial_fail_closed_and_original_200k_horizon():
    text = Path("scripts/run_mcvr_v8_multiseed_full.ps1").read_text(encoding="utf-8")
    seed12 = text.index("$Seed12 = Invoke-MCVRSeed -Seed 12")
    seed48 = text.index("$Seed48 = Invoke-MCVRSeed -Seed 48")
    report = text.index('"scripts/report_mcvr_v8_multiseed.py"')
    assert seed12 < seed48 < report
    assert "--steps 200000" in text
    assert "--steps 12500" not in text
    assert "--validation-batches 625" in text
    assert "MCVR_V8_MULTI_SEED_COMPLETED" in text
    assert "Remove-Item -LiteralPath $OutputDir" not in text
    assert "Test-PristineLauncherDirectory" in text
    assert '@("stdout.log", "stderr.log")' in text
    assert "Working tree must be clean" in text
    assert "formal_test_records_read" in text
    assert "frozen_holdout_records_read" in text
