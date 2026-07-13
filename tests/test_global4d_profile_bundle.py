import json
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from etflow.commons import global4d_profile_bundle as bundle
from scripts import print_global4d_profile_commands as command_printer


def _manifest():
    molecules = ["mol-a", "mol-a", "mol-a", "mol-b", "mol-b", "mol-c"]
    return {
        "manifest_version": "1.0",
        "records": [
            {
                "mol_id": molecule,
                "sample_id": f"sample-{index}",
                "x_init_hash": f"hash-{index}",
                "num_rotatable_bonds": index % 3,
            }
            for index, molecule in enumerate(molecules)
        ],
    }


class _Dataset:
    def __init__(self, root, split):
        root = Path(root)
        directory = root / split if (root / split).is_dir() else root
        self.data_files = sorted(directory.glob("*.pt"))

    def get(self, index):
        record = torch.load(self.data_files[index], map_location="cpu", weights_only=False)
        return SimpleNamespace(
            mol_id=record["mol_id"],
            source_mol_id=record["mol_id"],
            sample_id=record["sample_id"],
            x_init_hash=record["x_init_hash"],
            atomic_numbers=record["atomic_numbers"],
            num_rotatable_bonds=torch.tensor([record["num_rotatable_bonds"]]),
        )


def _validate(data, manifest):
    by_id = {item.sample_id: item for item in data}
    for row in manifest["records"]:
        item = by_id[row["sample_id"]]
        assert item.mol_id == row["mol_id"]
        assert item.x_init_hash == row["x_init_hash"]
    return by_id


def _sources(tmp_path, *, sensitive_config=False):
    source = tmp_path / "source"
    cache = source / "cache" / "test"
    cache.mkdir(parents=True)
    manifest = _manifest()
    for index, row in enumerate(manifest["records"]):
        torch.save(
            {
                **row,
                "atomic_numbers": torch.tensor([6, 6, 8]),
                "x_init": torch.full((3, 3), float(index)),
                "extra_marker": f"only-{index}",
            },
            cache / f"physical_{5-index:02d}.pt",
        )
    manifest_path = source / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    checkpoint = source / "step1000.ckpt"
    torch.save({"state_dict": {"weight": torch.arange(3)}}, checkpoint)
    config = source / "config.resolved.yaml"
    config_value = {
        "data": {"cache_dir": "/linux/source/cache"},
        "model": {},
        "loss": {},
        "optimizer": {"scheduler": "none"},
        "time_sampling": {},
    }
    if sensitive_config:
        config_value["wandb_api_key"] = "must-not-export"
    config.write_text(yaml.safe_dump(config_value), encoding="utf-8")
    return {
        "checkpoint": checkpoint,
        "config": config,
        "cache_dir": source / "cache",
        "manifest": manifest_path,
        "manifest_value": manifest,
    }


def _export(tmp_path, **overrides):
    sources = _sources(tmp_path)
    output = tmp_path / "artifacts" / "bundle.tar.gz"
    arguments = {
        **sources,
        "split": "test",
        "output": output,
        "max_molecules": 2,
        "max_records": 4,
        "seed": 42,
        "manifest_loader": lambda path: json.loads(Path(path).read_text()),
        "dataset_factory": _Dataset,
        "dataset_validator": _validate,
        "verification_callback": lambda root: bundle.verify_bundle_directory(
            root, verify_model=False, verify_dataset=False
        ),
    }
    arguments.pop("manifest_value")
    arguments.update(overrides)
    result = bundle.create_profile_bundle(**arguments)
    return sources, output, result


def _extract(archive, root):
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as handle:
        handle.extractall(root)
    return root / bundle.BUNDLE_ROOT_NAME


def test_deterministic_selection_same_seed_and_original_record_order():
    manifest = _manifest()
    first, first_report = bundle.select_manifest_records(
        manifest, max_molecules=2, max_records=4, seed=42
    )
    second, second_report = bundle.select_manifest_records(
        manifest, max_molecules=2, max_records=4, seed=42
    )
    assert first == second
    assert first_report == second_report
    source_order = [row["sample_id"] for row in manifest["records"]]
    selected_order = [row["sample_id"] for row in first]
    assert selected_order == [item for item in source_order if item in selected_order]


def test_max_molecules_and_max_records_are_enforced():
    selected, report = bundle.select_manifest_records(
        _manifest(), max_molecules=1, max_records=2, seed=42
    )
    assert len({row["mol_id"] for row in selected}) <= 1
    assert len(selected) <= 2
    assert report["selected_record_count"] == len(selected)


def test_bundle_copies_only_manifest_enabled_cache_and_uses_relative_paths(tmp_path):
    _, archive, result = _export(tmp_path)
    root = _extract(archive, tmp_path / "unpacked")
    manifest = json.loads((root / "manifest/profile_manifest.json").read_text())
    cache_files = list((root / "cache/test").glob("*.pt"))
    assert len(cache_files) == result["selected_record_count"]
    assert {row["cache_file"] for row in manifest["records"]} == {
        path.relative_to(root).as_posix() for path in cache_files
    }
    assert all(not Path(row["cache_file"]).is_absolute() for row in manifest["records"])


def test_hash_verification_succeeds_then_detects_tampering(tmp_path):
    _, archive, _ = _export(tmp_path)
    root = _extract(archive, tmp_path / "unpacked")
    assert bundle.verify_bundle_directory(
        root, verify_model=False, verify_dataset=False
    )["status"] == "VALID"
    cache_file = next((root / "cache/test").glob("*.pt"))
    with cache_file.open("ab") as handle:
        handle.write(b"tampered")
    result = bundle.verify_bundle_directory(
        root, verify_model=False, verify_dataset=False
    )
    assert result["status"] == "INVALID"
    assert "mismatch" in " ".join(result["errors"]).lower()


@pytest.mark.parametrize("unsafe", ["../escape.pt", "/linux/root.pt", "C:\\secret\\key.pt"])
def test_path_traversal_absolute_and_windows_drive_are_rejected(unsafe):
    with pytest.raises(bundle.BundleValidationError):
        bundle.safe_relative_path(unsafe)


def test_checkpoint_loads_on_cpu(tmp_path):
    checkpoint = tmp_path / "model.ckpt"
    torch.save({"state_dict": {"weight": torch.ones(2)}}, checkpoint)
    payload = bundle.load_checkpoint_cpu(checkpoint)
    assert payload["state_dict"]["weight"].device.type == "cpu"


def test_tar_member_names_are_cross_platform_and_non_executable(tmp_path):
    _, archive, _ = _export(tmp_path)
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
    assert all("\\" not in member.name and not PurePathDrive(member.name) for member in members)
    assert all(not member.issym() and not member.islnk() for member in members)
    assert all((member.mode & 0o111) == 0 for member in members if member.isfile())


def PurePathDrive(value):
    return len(value) >= 2 and value[1] == ":"


def test_force_false_does_not_overwrite(tmp_path):
    sources = _sources(tmp_path)
    output = tmp_path / "exists.tar.gz"
    output.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        bundle.create_profile_bundle(
            checkpoint=sources["checkpoint"],
            config=sources["config"],
            cache_dir=sources["cache_dir"],
            manifest=sources["manifest"],
            split="test",
            output=output,
        )
    assert output.read_bytes() == b"original"


def test_atomic_archive_failure_leaves_no_final_file(tmp_path, monkeypatch):
    sources = _sources(tmp_path)
    output = tmp_path / "failed.tar.gz"
    monkeypatch.setattr(bundle, "_tar_directory", lambda *args: (_ for _ in ()).throw(RuntimeError("tar failed")))
    with pytest.raises(RuntimeError, match="tar failed"):
        bundle.create_profile_bundle(
            checkpoint=sources["checkpoint"],
            config=sources["config"],
            cache_dir=sources["cache_dir"],
            manifest=sources["manifest"],
            split="test",
            output=output,
            max_molecules=1,
            max_records=1,
            manifest_loader=lambda path: json.loads(Path(path).read_text()),
            dataset_factory=_Dataset,
            dataset_validator=_validate,
            verification_callback=lambda root: bundle.verify_bundle_directory(
                root, verify_model=False, verify_dataset=False
            ),
        )
    assert not output.exists()
    assert not list(tmp_path.glob("failed.tar.gz.tmp.*"))


def test_optional_files_are_absent_by_default_and_no_sensitive_names_exist(tmp_path):
    _, archive, _ = _export(tmp_path)
    with tarfile.open(archive, "r:gz") as handle:
        names = handle.getnames()
    assert not any(name.endswith("partial_samples.pt") for name in names)
    assert not any(name.endswith("sampling_state.json") for name in names)
    forbidden = ("id_rsa", ".ssh", "wandb", "token", "private_key")
    assert not any(any(word in name.lower() for word in forbidden) for name in names)


def test_optional_files_are_included_only_when_explicitly_requested(tmp_path):
    state = tmp_path / "sampling_state.json"
    state.write_text(json.dumps({"completed_count": 2}), encoding="utf-8")
    partial = tmp_path / "partial_samples.pt"
    torch.save({"partial": True, "records": []}, partial)
    _, archive, _ = _export(
        tmp_path,
        include_sampling_state=True,
        sampling_state=state,
        include_partial_samples=True,
        partial_samples=partial,
    )
    with tarfile.open(archive, "r:gz") as handle:
        names = handle.getnames()
    assert any(name.endswith("optional/sampling_state.json") for name in names)
    assert any(name.endswith("optional/partial_samples.pt") for name in names)


def test_metadata_and_selection_report_have_required_counts_and_versions(tmp_path):
    _, archive, result = _export(tmp_path)
    root = _extract(archive, tmp_path / "unpacked")
    metadata = json.loads((root / "metadata/bundle_metadata.json").read_text())
    selection = json.loads((root / "metadata/selection_report.json").read_text())
    environment = json.loads((root / "metadata/environment_source.json").read_text())
    required = {
        "bundle_format_version",
        "source_git_commit",
        "source_branch",
        "source_hostname",
        "source_platform",
        "source_python_version",
        "source_torch_version",
        "source_cuda_version",
        "source_pyg_version",
        "source_rdkit_version",
        "checkpoint_sha256",
        "config_sha256",
        "original_manifest_sha256",
        "reduced_manifest_sha256",
        "selected_molecule_ids",
        "selected_record_ids",
        "exported_at",
        "creation_command",
    }
    assert required.issubset(metadata)
    assert selection["physical_copied_record_count"] == result["selected_record_count"]
    assert selection["manifest_enabled_record_count"] == result["selected_record_count"]
    bundled_config = yaml.safe_load((root / metadata["paths"]["config"]).read_text())
    assert bundled_config["data"]["cache_dir"] == "cache"
    assert set(environment) == {
        "python_version",
        "torch_version",
        "cuda_version",
        "pyg_version",
        "rdkit_version",
    }


def test_model_verification_instantiates_and_strictly_loads_state(monkeypatch):
    calls = {}

    class FakeModel:
        def __init__(self, **kwargs):
            calls["arguments"] = kwargs
            self._parameter = torch.nn.Parameter(torch.ones(2))

        def load_state_dict(self, state, strict):
            calls["state"] = state
            calls["strict"] = strict

        def parameters(self):
            return [self._parameter]

    fake = type(sys)("etflow.models.global_coupled_4d_flow")
    fake.GlobalCoupled4DFlowLightningModule = FakeModel
    monkeypatch.setitem(sys.modules, "etflow.models.global_coupled_4d_flow", fake)
    result = bundle._verify_model_and_checkpoint(
        {
            "model": {"hidden_dim": 8},
            "loss": {"final_weight": 1.0},
            "optimizer": {"lr": 0.001, "scheduler": "none"},
            "time_sampling": {"t_min": 0.0, "t_max": 0.25},
        },
        {"state_dict": {"weight": torch.ones(2)}},
    )
    assert calls["strict"] is True
    assert "scheduler" not in calls["arguments"]
    assert result["parameter_count"] == 2


def test_sensitive_config_is_rejected(tmp_path):
    sources = _sources(tmp_path, sensitive_config=True)
    with pytest.raises(bundle.BundleValidationError, match="Sensitive"):
        bundle.create_profile_bundle(
            checkpoint=sources["checkpoint"],
            config=sources["config"],
            cache_dir=sources["cache_dir"],
            manifest=sources["manifest"],
            split="test",
            output=tmp_path / "sensitive.tar.gz",
            max_molecules=1,
            max_records=1,
            manifest_loader=lambda path: json.loads(Path(path).read_text()),
            dataset_factory=_Dataset,
            dataset_validator=_validate,
            verification_callback=lambda root: {"status": "VALID"},
        )


def test_command_printer_only_prints_bounded_commands(tmp_path, monkeypatch, capsys):
    _, archive, _ = _export(tmp_path)
    root = _extract(archive, tmp_path / "unpacked")
    monkeypatch.setattr(sys, "argv", ["print_global4d_profile_commands.py", "--bundle_dir", str(root)])
    command_printer.main()
    output = capsys.readouterr().out
    assert "--max_records 20" in output
    assert "--max_records 30" in output
    assert "--disable_partial_save" in output
    assert "benchmark_global4d_sampling_io.py" in output
    assert "subprocess" not in output


def test_export_and_print_scripts_contain_no_training_sampling_or_eval_launch():
    for path in (
        Path("scripts/export_global4d_profile_bundle.py"),
        Path("scripts/verify_global4d_profile_bundle.py"),
        Path("scripts/print_global4d_profile_commands.py"),
    ):
        text = path.read_text(encoding="utf-8")
        assert "trainer.fit(" not in text
        assert "model.refine(" not in text
        assert "subprocess.run(" not in text
