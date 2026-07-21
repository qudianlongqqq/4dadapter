import json

from tests.external_refinement_test_utils import ROOT


def test_raw_smoke_evaluator_is_identity():
    path = ROOT / "diagnostics/ecir_mvr/external_refinement_baselines/formal_large_seed43/raw/smoke100/evaluation.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    for metric in ("weighted_bac_delta", "bond_delta", "angle_delta", "ring_delta", "mean_displacement"):
        assert report["metrics"][metric] == 0.0
    assert report["records"] == 100
