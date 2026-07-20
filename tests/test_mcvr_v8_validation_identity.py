import pytest

from etflow.ecir.v8_validation_cache import (
    new_prediction_manifest,
    prediction_identity,
    validate_manifest_identity,
)


def _identity(checkpoint="checkpoint"):
    return prediction_identity(
        checkpoint_sha256=checkpoint,
        config_sha256="config",
        validation_sources_sha256="sources",
        validation_targets_sha256="targets",
        record_ids=["a"],
        evaluator_semantics_sha256="evaluator",
        safety_semantics_sha256="safety",
    )


def test_checkpoint_identity_change_fails_closed(tmp_path):
    manifest = new_prediction_manifest(_identity(), chunk_size=1, output_dir=tmp_path)
    with pytest.raises(RuntimeError, match="identity changed"):
        validate_manifest_identity(manifest, _identity("other"))
