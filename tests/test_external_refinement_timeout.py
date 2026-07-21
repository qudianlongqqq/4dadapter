import subprocess

from etflow.ecir.external_refinement_baselines import refine_with_gfn2_xtb
from tests.external_refinement_test_utils import config, synthetic_record


def test_xtb_timeout_falls_back(monkeypatch, tmp_path):
    record, source = synthetic_record()
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 0.01)
    monkeypatch.setattr(subprocess, "run", timeout)
    result = refine_with_gfn2_xtb(record, source, config()["gfn2_xtb"], tmp_path / "isolated")
    assert result.timeout and result.fallback_to_source and result.failure_reason == "timeout"
