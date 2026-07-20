from etflow.ecir.v8_validation_cache import prediction_identity


def test_baseline_method_and_checkpoint_are_identity_bound():
    common = dict(
        config_sha256="config",
        validation_sources_sha256="sources",
        validation_targets_sha256="targets",
        record_ids=["a", "b"],
        evaluator_semantics_sha256="evaluator",
        safety_semantics_sha256="safety",
    )
    d1 = prediction_identity(method="D1", checkpoint_sha256="d1", **common)
    v7 = prediction_identity(method="V7", checkpoint_sha256="d1", **common)
    assert d1["identity_sha256"] != v7["identity_sha256"]
    assert d1["record_identity_sha256"] == v7["record_identity_sha256"]
