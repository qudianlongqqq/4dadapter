import json
import sys
from types import SimpleNamespace

import torch

from etflow.data.flexbond_eval_manifest import build_manifest_aware_sample_payload
from scripts import eval_flexbond_optimizer as shared_eval
from scripts import eval_global_coupled_4d_flow as global_eval


def test_global4d_payload_is_accepted_through_wrapper_and_shared_evaluator(
    tmp_path, monkeypatch
):
    manifest = {
        "manifest_version": "1.0",
        "created_at": "2026-07-12T00:00:00+00:00",
        "records": [
            {
                "mol_id": "molecule",
                "sample_id": "sample",
                "x_init_hash": "x-init-hash",
                "num_rotatable_bonds": 1,
            },
            {
                "mol_id": "molecule-2",
                "sample_id": "sample-2",
                "x_init_hash": "x-init-hash-2",
                "num_rotatable_bonds": 1,
            },
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    x_init = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    inference_record = SimpleNamespace(
        mol_id="sample",
        source_mol_id="molecule",
        sample_id="sample",
        x_init_hash="x-init-hash",
        num_rotatable_bonds=torch.tensor([1]),
        x_init=x_init,
    )
    inference_record_2 = SimpleNamespace(
        mol_id="sample-2",
        source_mol_id="molecule-2",
        sample_id="sample-2",
        x_init_hash="x-init-hash-2",
        num_rotatable_bonds=torch.tensor([1]),
        x_init=x_init,
    )
    inference = {"sample": inference_record, "sample-2": inference_record_2}
    records = [{
        "mol_id": "sample",
        "source_mol_id": "molecule",
        "sample_id": "sample",
        "method_name": "global_coupled_4d_adapter",
        "x_init_hash": "x-init-hash",
        "status": "success",
        "x_refined": x_init.clone(),
        "alpha": 0.5,
    }]
    payload = build_manifest_aware_sample_payload(
        records=records,
        manifest=manifest,
        manifest_path=tmp_path / "sampler-side-manifest-alias.json",
        selected_manifest={**manifest, "records": manifest["records"][:1]},
        split="test",
        inference_cache_path=tmp_path / "sampler-cache-alias",
        inference_by_id=inference,
        extra={"failure_count": 0, "failure_rate": 0.0},
    )
    samples = tmp_path / "global4d_samples.pt"
    torch.save(payload, samples)

    reference = SimpleNamespace(
        mol_id="sample",
        reference_conformer_ptr=torch.tensor([0, 2]),
        x_ref_candidates=x_init.clone(),
    )
    reference_2 = SimpleNamespace(
        mol_id="sample-2",
        reference_conformer_ptr=torch.tensor([0, 2]),
        x_ref_candidates=x_init.clone(),
    )
    monkeypatch.setattr(
        shared_eval,
        "FlexBondInferenceDataset",
        lambda cache, split: [inference_record, inference_record_2],
    )
    monkeypatch.setattr(
        shared_eval,
        "FlexBondOptimizerDataset",
        lambda cache, split, validate: [reference, reference_2],
    )
    output = tmp_path / "evaluation"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_global_coupled_4d_flow.py",
            "--manifest", str(manifest_path),
            "--inference_cache", str(tmp_path / "evaluator-cache-alias"),
            "--reference_cache", str(tmp_path / "references"),
            "--split", "test",
            "--samples", str(samples),
            "--output_dir", str(output),
        ],
    )
    global_eval.main()

    assert (output / "summary.csv").is_file()
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["diagnostics"]["global_coupled_4d_adapter"]["missing_ids"] == [
        "sample-2"
    ]
    assert (output / "COMPLETED").is_file()
