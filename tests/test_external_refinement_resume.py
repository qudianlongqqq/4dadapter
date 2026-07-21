from etflow.ecir.v8_validation_cache import append_prediction_chunk, completed_chunk_ranges, new_prediction_manifest


def test_resume_recognizes_completed_ranges(tmp_path):
    identity = {"identity_sha256": "x", "record_count": 1}
    manifest = new_prediction_manifest(identity, chunk_size=1, output_dir=tmp_path)
    path = tmp_path / "prediction_manifest.json"
    append_prediction_chunk(path, manifest, record_start=0, record_end=1, records=[{"record_index": 0}])
    assert completed_chunk_ranges(manifest) == {(0, 1)}
