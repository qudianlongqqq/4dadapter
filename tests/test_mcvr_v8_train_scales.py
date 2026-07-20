import hashlib
import json

import pytest

from etflow.ecir.v8_constraint_normalization import FrozenResidualScales


def test_scales_reject_validation_or_test(tmp_path):
    payload = {
        "split": "train",
        "scales": {"bond": 0.1, "angle": 0.2, "clash": 0.3, "ring": 0.4, "chirality": 0.5},
        "validation_used": False,
        "test_used": False,
        "frozen_holdout_used": False,
    }
    payload["identity_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path = tmp_path / "scales.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert FrozenResidualScales.load(path).bond == 0.1
    payload["validation_used"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError):
        FrozenResidualScales.load(path)
