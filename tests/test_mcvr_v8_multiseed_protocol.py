import json

import pandas as pd
import pytest

from scripts.preflight_mcvr_v8_multiseed import build_registry
from scripts.report_mcvr_v8_multiseed import METRICS, build_summary


def _evaluation(path, seed):
    value = float(seed)
    payload = {
        "mode": "FULL",
        "records": 10000,
        "formal_test_records_read": 0,
        "frozen_holdout_records_read": 0,
        "metrics": {
            "weighted_bac_delta": value,
            "bond_delta": value + 1,
            "angle_delta": value + 2,
            "active_angle_delta": value + 2.5,
            "ring_delta": value + 3,
            "clash_delta": value + 4,
            "accepted": value / 100,
            "mean_displacement": value / 1000,
            "rmsd": value / 10,
        },
        "set_metrics": {
            "COV_P": value + 5,
            "COV_R": value + 6,
            "MAT_P": value + 7,
            "MAT_R": value + 8,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_registry_freezes_identities_outputs_metrics_and_isolation():
    registry = build_registry(require_clean=False)
    assert registry["frozen_base_git_sha"] == "4df21d766afadab169ecc7208477a6ca6ffe384a"
    assert registry["new_training_seeds"] == [12, 48]
    assert registry["training_contract"]["total_record_exposure"] == 800000
    assert registry["analysis_protocol"]["ddof"] == 1
    assert registry["analysis_protocol"]["report_both_cov_mat_directions"] is True
    assert registry["analysis_protocol"]["single_direction_posthoc_selection"] is False
    assert registry["analysis_protocol"]["clash_interpretation"] == "low-power natural cohort"
    assert registry["identities"]["d1_checkpoint_sha256"] == (
        "c7f2e5e36a400600951d846b7d11d1d9aa57a0da78d2e540340fe44b470868ca"
    )
    assert registry["isolation"]["formal_test_records_read"] == 0
    assert registry["isolation"]["frozen_holdout_records_read"] == 0
    for seed in ("12", "48"):
        assert len(registry["configs"][seed]["config_file_sha256"]) == 64
        assert len(registry["configs"][seed]["inherited_resolved_config_sha256"]) == 64


def test_summary_uses_three_seeds_sample_std_and_all_cov_mat_directions(tmp_path):
    paths = {}
    for seed in (12, 43, 48):
        path = tmp_path / f"seed{seed}.json"
        _evaluation(path, seed)
        paths[seed] = path
    frame, payload = build_summary(paths)
    numeric = pd.Series([12.0, 43.0, 48.0])
    assert payload["mean"]["weighted_bac"] == pytest.approx(numeric.mean())
    assert payload["sample_std_ddof1"]["weighted_bac"] == pytest.approx(numeric.std(ddof=1))
    assert payload["cov_mat_reporting"] == ["COV_P", "COV_R", "MAT_P", "MAT_R"]
    assert payload["metric_mapping"]["active_angle"] == "active_angle_delta"
    assert set(METRICS).issubset(frame.columns)
    assert frame.seed.tolist() == [12, 43, 48, "mean", "sample_std_ddof1"]


def test_summary_fails_closed_on_test_or_holdout_reads(tmp_path):
    paths = {}
    for seed in (12, 43, 48):
        path = tmp_path / f"seed{seed}.json"
        _evaluation(path, seed)
        paths[seed] = path
    payload = json.loads(paths[48].read_text(encoding="utf-8"))
    payload["formal_test_records_read"] = 1
    paths[48].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="formal test"):
        build_summary(paths)
