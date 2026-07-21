import copy
import inspect
import random
from itertools import islice

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

import scripts.train_ecir_mvr_v8 as runner
from etflow.ecir.v8_sampler import sampler_from_payload


SEEDS = (12, 43, 48)
CONFIGS = {
    12: "configs/ecir_mvr_v8_full_v1_formal_large_200k_seed12.yaml",
    48: "configs/ecir_mvr_v8_full_v1_formal_large_200k_seed48.yaml",
}


class WorkerSeedProbe(Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, index):
        return torch.tensor(
            [
                int(index),
                int(torch.initial_seed() % 2**31),
                int(random.randrange(2**30)),
                int(np.random.randint(0, 2**30)),
                int(torch.randint(0, 2**30, ()).item()),
            ],
            dtype=torch.int64,
        )


def _sampler_payload():
    return {
        "split": "train",
        "test_used": False,
        "records": [
            {"sampling_weight": value}
            for value in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        ],
    }


def _sampler_trace(seed, count=32):
    return list(sampler_from_payload(_sampler_payload(), num_samples=count, seed=seed))


def _rng_trace(seed):
    runner._seed(seed)
    result = {
        "python": [random.random() for _ in range(8)],
        "numpy": np.random.random(8).tolist(),
        "torch_cpu": torch.rand(8).tolist(),
        "sampler": _sampler_trace(seed),
    }
    if torch.cuda.is_available():
        result["cuda"] = torch.rand(8, device="cuda").cpu().tolist()
    return result


def _worker_trace(seed):
    runner._seed(seed)
    loader = DataLoader(WorkerSeedProbe(), batch_size=2, num_workers=2, shuffle=False)
    return torch.cat(list(loader), dim=0)


def _short_training_trajectory(global_seed, dataset_seed):
    """Small deterministic optimization fingerprint for legacy/new Seed43 parity."""

    runner._seed(global_seed)
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-6)
    indices = _sampler_trace(global_seed, count=6)
    losses = []
    for index in indices:
        generator = torch.Generator().manual_seed(dataset_seed + int(index))
        features = torch.rand((4, 3), generator=generator)
        target = torch.rand((4, 1), generator=generator)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(features), target)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return {
        "indices": indices,
        "losses": losses,
        "state": {name: value.detach().clone() for name, value in model.state_dict().items()},
    }


def _drop_paths(payload, paths):
    result = copy.deepcopy(payload)
    for path in paths:
        cursor = result
        for key in path[:-1]:
            cursor = cursor[key]
        cursor.pop(path[-1], None)
    return result


def test_seed43_fix_is_short_trajectory_equivalent_to_legacy_hardcoded_43():
    legacy = _short_training_trajectory(global_seed=43, dataset_seed=43)
    resolved_seed = int(
        runner.load_config("configs/ecir_mvr_v8_full_v1_formal_large_200k.yaml")["seed"]
    )
    repaired = _short_training_trajectory(global_seed=resolved_seed, dataset_seed=resolved_seed)
    assert legacy["indices"] == repaired["indices"]
    assert legacy["losses"] == repaired["losses"]
    for name in legacy["state"]:
        assert torch.equal(legacy["state"][name], repaired["state"][name])


def test_seed12_seed43_seed48_have_distinct_rng_and_sampler_trajectories():
    traces = {seed: _rng_trace(seed) for seed in SEEDS}
    for left, right in ((12, 43), (12, 48), (43, 48)):
        assert traces[left]["python"] != traces[right]["python"]
        assert traces[left]["numpy"] != traces[right]["numpy"]
        assert traces[left]["torch_cpu"] != traces[right]["torch_cpu"]
        assert traces[left]["sampler"] != traces[right]["sampler"]
        if torch.cuda.is_available():
            assert traces[left]["cuda"] != traces[right]["cuda"]


@pytest.mark.parametrize("seed", SEEDS)
def test_same_seed_reproduces_all_rng_sampler_and_dataloader_worker_sequences(seed):
    assert _rng_trace(seed) == _rng_trace(seed)
    assert torch.equal(_worker_trace(seed), _worker_trace(seed))


def test_dataloader_worker_sequences_differ_across_seeds():
    assert not torch.equal(_worker_trace(12), _worker_trace(48))


def test_dataset_receives_resolved_seed(monkeypatch):
    captured = []

    class FakeDataset:
        def __init__(self, *args, seed, **kwargs):
            captured.append(seed)
            self.sources = pd.DataFrame(
                {
                    "generator_name": ["source"],
                    "source_severity": ["normal"],
                }
            )
            self.plan = []

    monkeypatch.setattr(runner, "MCVRMixedDataset", FakeDataset)
    monkeypatch.setattr(runner.pd, "read_parquet", lambda path: pd.DataFrame({"row": [0]}))
    for seed in SEEDS:
        dataset = runner._real_dataset(
            runner.Path("sources.parquet"),
            runner.Path("targets.parquet"),
            object(),
            seed=seed,
            source_cache_root=None,
            target_cache_root=None,
            source_identity="identity",
        )
        assert dataset.plan[0]["sample_type"] == "real_error"
    assert captured == list(SEEDS)
    assert "seed=43" not in inspect.getsource(runner._real_dataset)


def test_rng_sampler_and_exposure_are_continuous_across_resume():
    seed = 12
    runner._seed(seed)
    full_sampler = _sampler_trace(seed, count=24)
    prefix_count = 9
    python_prefix = [random.random() for _ in range(5)]
    numpy_prefix = np.random.random(5)
    torch_prefix = torch.rand(5)
    checkpoint = {
        "rng_states": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        "sampler_state": {
            "global_step": 3,
            "effective_batch": 3,
            "records_exposed": prefix_count,
            "next_record_exposure_offset": prefix_count,
        },
    }
    expected_rng = (
        [random.random() for _ in range(5)],
        np.random.random(5),
        torch.rand(5),
    )
    runner._seed(999)
    runner._restore_rng_states(checkpoint)
    resumed_rng = (
        [random.random() for _ in range(5)],
        np.random.random(5),
        torch.rand(5),
    )
    assert expected_rng[0] == resumed_rng[0]
    assert np.array_equal(expected_rng[1], resumed_rng[1])
    assert torch.equal(expected_rng[2], resumed_rng[2])
    resumed_sampler = list(islice(iter(_sampler_trace(seed, count=24)), prefix_count, None))
    assert full_sampler[prefix_count:] == resumed_sampler
    assert checkpoint["sampler_state"]["records_exposed"] == prefix_count
    assert checkpoint["sampler_state"]["next_record_exposure_offset"] == prefix_count
    assert len(python_prefix) == len(numpy_prefix) == len(torch_prefix) == 5


def test_multiseed_configs_only_change_preregistered_identity_fields():
    frozen = runner.load_config("configs/ecir_mvr_v8_full_v1_formal_large_200k.yaml")
    allowed = {
        ("experiment_name",),
        ("seed",),
        ("long_run", "parent_5k_checkpoint"),
        ("long_run", "parent_5k_checkpoint_sha256"),
        ("long_run", "resume_audit_required"),
        ("long_run", "start_step"),
        ("multiseed_registration",),
    }
    for seed, path in CONFIGS.items():
        candidate = runner.load_config(path)
        assert candidate["seed"] == seed
        assert candidate["steps_total"] == 200000
        assert candidate["training"]["optimizer_steps"] == 200000
        assert candidate["training"]["effective_batch_size"] == 64
        assert (
            candidate["training"]["batch_size"]
            * candidate["training"]["gradient_accumulation_steps"]
            == 64
        )
        assert candidate["model"]["d1_checkpoint_sha256"] == (
            "c7f2e5e36a400600951d846b7d11d1d9aa57a0da78d2e540340fe44b470868ca"
        )
        assert 10000 in candidate["validation_protocol"]["fast_steps"]
        assert candidate["isolation"] == runner.ISOLATION
        assert _drop_paths(candidate, allowed) == _drop_paths(frozen, allowed)


def test_registered_stop_request_exists_before_training_and_preserves_isolation(tmp_path):
    config = runner.load_config(CONFIGS[12])
    request = runner._materialize_registered_stop_request(
        tmp_path,
        config,
        planned_total_steps=200000,
        effective_batch=64,
    )
    assert request["request_origin"] == "resolved_config_before_first_optimizer_step"
    assert request["user_requested_stop_step"] == 12500
    assert request["total_record_exposure"] == 800000
    assert request["formal_test_records_read"] == 0
    assert request["frozen_holdout_records_read"] == 0
    assert runner._read_graceful_stop_request(
        tmp_path,
        current_step=0,
        planned_total_steps=200000,
        effective_batch=64,
    ) == request


def test_multiseed_configs_forbid_formal_test_and_frozen_holdout():
    assert runner.ISOLATION["formal_test_records_read"] == 0
    assert runner.ISOLATION["frozen_holdout_records_read"] == 0
    for path in CONFIGS.values():
        config = runner.load_config(path)
        assert config["data"]["allow_formal_test"] is False
        assert config["data"]["allow_frozen_holdout"] is False
        assert config["multiseed_registration"]["formal_test_records_read"] == 0
        assert config["multiseed_registration"]["frozen_holdout_records_read"] == 0
