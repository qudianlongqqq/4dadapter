import json

from etflow.ecir.external_refinement_baselines import ISOLATION
from tests.external_refinement_test_utils import ROOT, config


def test_formal_test_and_holdout_remain_isolated():
    assert config()["isolation"] == ISOLATION
    report = json.loads((ROOT / "reports/ecir_mvr/external_refinement_baselines/SMOKE100_RESULTS.json").read_text(encoding="utf-8"))
    for key, value in ISOLATION.items():
        assert report[key] == value
