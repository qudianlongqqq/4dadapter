import json

from etflow.ecir.v8_validation_cache import (
    append_prediction_chunk,
    completed_chunk_ranges,
    finish_prediction_manifest,
    new_prediction_manifest,
    prediction_identity,
)


def _identity():
    return prediction_identity(
        checkpoint_sha256="c",
        config_sha256="g",
        validation_sources_sha256="s",
        validation_targets_sha256="t",
        record_ids=["a", "b"],
        evaluator_semantics_sha256="e",
        safety_semantics_sha256="f",
    )


def test_chunk_resume_is_contiguous_and_atomic(tmp_path):
    path = tmp_path / "prediction_manifest.json"
    manifest = new_prediction_manifest(_identity(), chunk_size=1, output_dir=tmp_path)
    append_prediction_chunk(path, manifest, record_start=0, record_end=1, records=[{"id": "a"}])
    restored = json.loads(path.read_text())
    assert completed_chunk_ranges(restored) == {(0, 1)}
    append_prediction_chunk(path, manifest, record_start=1, record_end=2, records=[{"id": "b"}])
    finish_prediction_manifest(path, manifest)
    assert json.loads(path.read_text())["status"] == "COMPLETED"
