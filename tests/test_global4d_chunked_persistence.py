import json
from pathlib import Path

import pytest
import torch

from etflow.commons import global4d_chunked_persistence as chunks
from etflow.commons import global_coupled_4d_sampling as atomic_io


def _rows(count=12):
    return [
        {
            "mol_id": f"mol-{index // 3}",
            "sample_id": f"sample-{index:03d}",
            "x_init_hash": f"hash-{index:03d}",
            "num_rotatable_bonds": index % 4,
        }
        for index in range(count)
    ]


def _records(count=12):
    return [
        {
            **row,
            "source_mol_id": row["mol_id"],
            "x_refined": torch.full((3, 3), float(index)),
            "status": "success",
        }
        for index, row in enumerate(_rows(count))
    ]


def _identity(**changes):
    return {
        "checkpoint_inference_sha256": "checkpoint",
        "config_sha256": "config",
        "manifest_sha256": "manifest",
        "split": "test",
        "alpha": 0.2,
        **changes,
    }


def _write(root, records, rows, index, start, previous=None):
    return chunks.write_chunk(
        root,
        records=records,
        selected_rows=rows,
        chunk_index=index,
        start=start,
        run_identity=_identity(),
        previous_chunk_sha256=previous,
    )


def test_chunk_append_does_not_rewrite_validated_old_chunks(tmp_path):
    rows, records = _rows(), _records()
    first, first_hash, created = _write(tmp_path, records[:5], rows, 0, 0)
    first_bytes = first.read_bytes()
    _, second_hash, created_second = _write(
        tmp_path, records[5:10], rows, 1, 5, first_hash
    )
    assert created and created_second and second_hash
    assert first.read_bytes() == first_bytes
    same, same_hash, rewritten = _write(tmp_path, records[:5], rows, 0, 0)
    assert same == first and same_hash == first_hash and rewritten is False


def test_chunk_bytes_scale_linearly_and_merge_preserves_exact_order(tmp_path):
    rows, records = _rows(30), _records(30)
    scan = chunks.convert_records_to_chunks(
        records,
        tmp_path,
        selected_rows=rows,
        run_identity=_identity(),
        save_every_records=7,
    )
    assert scan.chunk_count == 5
    assert [row["sample_id"] for row in scan.records] == [
        row["sample_id"] for row in records
    ]
    assert scan.total_bytes < 2.5 * sum(
        path.stat().st_size for path in scan.chunk_paths[-2:]
    ) * (len(scan.chunk_paths) / 2)


def test_compact_state_has_no_growing_id_list_and_nearly_constant_size(tmp_path):
    common = dict(
        status="PARTIAL",
        total_count=1000,
        completed_chunk_count=1,
        current_chunk_size=0,
        save_every_records=50,
        run_identity=_identity(),
        ordered_sample_ids_hash="a" * 64,
        output=tmp_path / "samples.pt",
        device="cpu",
        started_at="2026-01-01T00:00:00+00:00",
        latest_chunk_sha256="b" * 64,
    )
    small = chunks.compact_sampling_state(completed_count=10, **common)
    large = chunks.compact_sampling_state(completed_count=999, **common)
    forbidden = {"completed_ordered_sample_ids", "records", "profile_rows"}
    assert forbidden.isdisjoint(small)
    assert abs(len(json.dumps(small)) - len(json.dumps(large))) < 8


def test_chunk_order_duplicate_and_manifest_interval_are_rejected(tmp_path):
    rows, records = _rows(), _records()
    _, first_hash, _ = _write(tmp_path, records[:4], rows, 0, 0)
    second, _, _ = _write(tmp_path, records[4:8], rows, 1, 4, first_hash)
    payload = torch.load(second, map_location="cpu", weights_only=False)
    payload["records"][0]["sample_id"] = records[0]["sample_id"]
    torch.save(payload, second)
    with pytest.raises(ValueError, match="manifest interval|content hash"):
        chunks.scan_chunks(tmp_path, selected_rows=rows, run_identity=_identity())


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("checkpoint_inference_sha256", "changed-checkpoint"),
        ("config_sha256", "changed-config"),
        ("manifest_sha256", "changed-manifest"),
    ],
)
def test_missing_chunk_and_changed_run_identity_are_rejected(
    tmp_path, field, changed
):
    rows, records = _rows(), _records()
    first, first_hash, _ = _write(tmp_path, records[:4], rows, 0, 0)
    _write(tmp_path, records[4:8], rows, 1, 4, first_hash)
    first.unlink()
    with pytest.raises(ValueError, match="numbering"):
        chunks.scan_chunks(tmp_path, selected_rows=rows, run_identity=_identity())
    other = tmp_path / "other"
    _write(other, records[:4], rows, 0, 0)
    with pytest.raises(ValueError, match="different sampling command"):
        chunks.scan_chunks(
            other,
            selected_rows=rows,
            run_identity=_identity(**{field: changed}),
        )


def test_hash_chain_detects_tampered_earlier_chunk(tmp_path):
    rows, records = _rows(), _records()
    first, first_hash, _ = _write(tmp_path, records[:4], rows, 0, 0)
    _write(tmp_path, records[4:8], rows, 1, 4, first_hash)
    payload = torch.load(first, map_location="cpu", weights_only=False)
    payload["unexpected"] = "tamper"
    torch.save(payload, first)
    with pytest.raises(ValueError, match="hash chain"):
        chunks.scan_chunks(tmp_path, selected_rows=rows, run_identity=_identity())


def test_copied_duplicate_chunk_is_rejected(tmp_path):
    rows, records = _rows(), _records()
    first, _, _ = _write(tmp_path, records[:4], rows, 0, 0)
    (tmp_path / "chunk_000001.pt").write_bytes(first.read_bytes())
    with pytest.raises(ValueError, match="index|contiguous"):
        chunks.scan_chunks(tmp_path, selected_rows=rows, run_identity=_identity())


def test_interrupted_atomic_chunk_is_not_seen_as_complete(tmp_path, monkeypatch):
    rows, records = _rows(), _records()

    def interrupted(payload, path):
        torch.save(payload, path.with_name(path.name + ".tmp.interrupted"))
        raise RuntimeError("power loss")

    monkeypatch.setattr(chunks, "atomic_torch_save", interrupted)
    with pytest.raises(RuntimeError, match="power loss"):
        _write(tmp_path, records[:4], rows, 0, 0)
    assert not (tmp_path / "chunk_000000.pt").exists()
    scan = chunks.scan_chunks(
        tmp_path, selected_rows=rows, run_identity=_identity()
    )
    assert scan.completed_count == 0


@pytest.mark.parametrize("kind", ["state", "final_merge"])
def test_state_or_final_merge_rename_interruption_preserves_durable_chunks(
    tmp_path, monkeypatch, kind
):
    rows, records = _rows(), _records()
    _write(tmp_path / "partial_chunks", records[:4], rows, 0, 0)
    destination = tmp_path / (
        "sampling_state.json" if kind == "state" else "samples.pt"
    )

    def interrupted(source, target):
        raise RuntimeError("simulated rename interruption")

    monkeypatch.setattr(atomic_io.os, "replace", interrupted)
    writer = (
        atomic_io.atomic_json_save
        if kind == "state"
        else atomic_io.atomic_torch_save
    )
    with pytest.raises(RuntimeError, match="rename interruption"):
        payload = {"completed_count": 4}
        if kind == "final_merge":
            payload["records"] = records[:4]
        writer(payload, destination)
    assert not destination.exists()
    scan = chunks.scan_chunks(
        tmp_path / "partial_chunks",
        selected_rows=rows,
        run_identity=_identity(),
    )
    assert scan.completed_count == 4


def test_state_may_lag_one_atomic_chunk_but_may_not_lead(tmp_path):
    rows, records = _rows(), _records()
    first, first_hash, _ = _write(tmp_path, records[:4], rows, 0, 0)
    state = chunks.compact_sampling_state(
        status="PARTIAL",
        completed_count=4,
        total_count=len(rows),
        completed_chunk_count=1,
        current_chunk_size=0,
        save_every_records=4,
        run_identity=_identity(),
        ordered_sample_ids_hash=chunks.ordered_sample_ids_sha256(
            [row["sample_id"] for row in rows]
        ),
        output=tmp_path / "samples.pt",
        device="cpu",
        started_at="2026-01-01T00:00:00+00:00",
        latest_chunk_sha256=first_hash,
    )
    _write(tmp_path, records[4:8], rows, 1, 4, first_hash)
    scan = chunks.scan_chunks(tmp_path, selected_rows=rows, run_identity=_identity())
    chunks.validate_compact_state(
        state,
        scan=scan,
        run_identity=_identity(),
        ordered_sample_ids_hash=state["ordered_sample_ids_sha256"],
        save_every_records=4,
    )
    state["completed_count"] = 9
    with pytest.raises(ValueError, match="ahead"):
        chunks.validate_compact_state(
            state,
            scan=scan,
            run_identity=_identity(),
            ordered_sample_ids_hash=state["ordered_sample_ids_sha256"],
            save_every_records=4,
        )


def test_legacy_record_conversion_is_idempotent_and_retains_source(tmp_path):
    rows, records = _rows(9), _records(9)
    legacy = tmp_path / "partial_samples.pt"
    torch.save({"partial": True, "records": records}, legacy)
    root = tmp_path / "partial_chunks"
    first = chunks.convert_records_to_chunks(
        records,
        root,
        selected_rows=rows,
        run_identity=_identity(),
        save_every_records=4,
    )
    original_hashes = list(first.chunk_hashes)
    second = chunks.convert_records_to_chunks(
        records,
        root,
        selected_rows=rows,
        run_identity=_identity(),
        save_every_records=4,
    )
    assert legacy.exists()
    assert second.chunk_hashes == original_hashes
    assert second.completed_count == len(records)


def test_chunk_root_and_files_must_not_be_symlinks(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("Symlink creation is unavailable")
    with pytest.raises(ValueError, match="symlink"):
        chunks.scan_chunks(link, selected_rows=_rows(), run_identity=_identity())
