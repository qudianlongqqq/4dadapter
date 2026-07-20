import torch

from etflow.ecir.mcvr_v8_full import MCVRV8FullRefiner
from etflow.ecir.mvr_model import MCVRModel


def test_checkpoint_roundtrip_preserves_unroll_and_state(tmp_path):
    prior = MCVRModel(
        hidden_dim=8,
        edge_hidden_dim=8,
        time_embedding_dim=4,
        num_layers=1,
        encoder_num_layers=1,
        error_embedding_dim=4,
    )
    model = MCVRV8FullRefiner(prior, unroll_steps=2)
    path = tmp_path / "v8.ckpt"
    torch.save(
        {
            "schema_version": "mcvr-v8-full-v1-checkpoint-v1",
            "unroll_steps": model.unroll_steps,
            "resolved_config_sha256": "config",
            "residual_scales_identity_sha256": "scales",
            "model_state_dict": model.state_dict(),
        },
        path,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    clone = MCVRV8FullRefiner(
        MCVRModel(
            hidden_dim=8,
            edge_hidden_dim=8,
            time_embedding_dim=4,
            num_layers=1,
            encoder_num_layers=1,
            error_embedding_dim=4,
        ),
        unroll_steps=payload["unroll_steps"],
    )
    incompatible = clone.load_state_dict(payload["model_state_dict"], strict=True)
    assert not incompatible.missing_keys and not incompatible.unexpected_keys
    assert payload["resolved_config_sha256"] == "config"
    assert payload["residual_scales_identity_sha256"] == "scales"
