import pytest

from etflow.data.flexbond_inference_dataset import validate_inference_record
from tests.test_flexbond_data_contract import _record


def test_inference_contract_rejects_reference_coordinates():
    record = _record()
    record.pop("x_ref_candidates")
    record["x_ref"] = record["x_init"]
    with pytest.raises(ValueError, match="forbidden label"):
        validate_inference_record(record)


def test_inference_contract_accepts_label_free_record():
    record = _record()
    for key in list(record):
        if key.startswith("x_ref") or key.startswith("selected_ref"):
            record.pop(key)
    record.pop("rmsd_before", None)
    record.pop("rmsd_after", None)
    checked = validate_inference_record(record)
    assert checked["x_init_hash"]
