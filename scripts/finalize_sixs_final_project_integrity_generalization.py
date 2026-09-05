#!/usr/bin/env python3
"""Low-memory finalizer for the read-only SIXS integrity audit.

The lineage, overlap and fingerprint stages are already frozen in reports
01--11.  This entry point deliberately does not reopen the large prepared
Reference cache; it reconstructs the DEV molecule table from existing frozen
artifacts and executes the unchanged report-12-through-status section of the
auditor.
"""

from __future__ import annotations

from audit_sixs_final_project_integrity_generalization import *


def main() -> None:
    RDLogger.DisableLog("rdApp.*")
    required = [OUT / name for name in REPORT_NAMES[:11]]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"cached audit stages 01--11 are incomplete: {missing}")

    manifest = json.loads(DEV_MANIFEST.read_text(encoding="utf-8"))
    dev_ids = {sample_id for row in manifest["rows"] for sample_id in row["sample_ids"]}
    val = pd.read_parquet(
        VAL,
        columns=["sample_id", "molecule_id", "num_atoms", "num_rotatable_bonds", "generator_name"],
    )
    dev = val[val.sample_id.isin(dev_ids)].copy()
    del val
    rec, mol = build_metrics(dev)

    sim = pd.read_csv(OUT / "10_TRAIN_DEV_SIMILARITY.csv")
    mol = mol.merge(
        sim[["molecule_id", "max_train_similarity", "nearest_train_molecule_id", "scaffold_seen_in_train"]],
        on="molecule_id", how="left", validate="one_to_one",
    )
    mol["similarity_quintile"] = equal_quintile(mol.max_train_similarity)

    overlap = pd.read_csv(OUT / "06_EXACT_SPLIT_OVERLAP.csv")
    constants = pd.read_csv(OUT / "03_HUMAN_CONSTANTS.csv")
    for column in ("ACTIVE_IN_RESTRICTED", "ACTIVE_IN_UNRESTRICTED"):
        constants[column] = constants[column].astype(str).str.lower().map({"true": True, "false": False})
    r_human = int(((constants.CLASS.eq("HUMAN_SCIENTIFIC_DESIGN")) & constants.ACTIVE_IN_RESTRICTED).sum())
    u_human = int(((constants.CLASS.eq("HUMAN_SCIENTIFIC_DESIGN")) & constants.ACTIVE_IN_UNRESTRICTED).sum())
    questionable = constants[
        constants.CLASS.eq("HUMAN_SCIENTIFIC_DESIGN")
        & constants.CURRENT_JUSTIFICATION.str.contains("not |weak|historical|design choice", case=False, regex=True)
    ].CONSTANT.tolist()

    required_fragments = ["belief.geometry", "reliability", "adaptive_ba", "magnitude"]
    r_ck = torch.load(RESTRICTED_CHECKPOINT, map_location="cpu", weights_only=False)
    u_ck = torch.load(UNRESTRICTED_CHECKPOINT, map_location="cpu", weights_only=False)
    r_keys, u_keys = set(r_ck["model_state"]), set(u_ck["model_state"])
    r_identity = all(any(fragment in key for key in r_keys) for fragment in required_fragments) and r_ck["step"] == 17500 and r_ck["config_sha256"] == sha256(RESTRICTED_CONFIG)
    u_identity = all(any(fragment in key for key in u_keys) for fragment in required_fragments) and u_ck["step"] == 17500 and u_ck["config_sha256"] == sha256(UNRESTRICTED_CONFIG)
    del r_ck, u_ck

    decisions = pd.DataFrame(index=range(9))
    train_unique_count = 50000
    dev_unique_count = int(dev.molecule_id.nunique())
    train_fps = [None] * train_unique_count  # only len() is used by FINAL_STATUS

    source = Path(__file__).with_name("audit_sixs_final_project_integrity_generalization.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    main_node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    start = next(
        i for i, node in enumerate(main_node.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "outcomes" for target in node.targets)
    )
    continuation = ast.fix_missing_locations(ast.Module(body=main_node.body[start:], type_ignores=[]))
    namespace = dict(globals())
    namespace.update(locals())
    exec(
        compile(continuation, str(Path(__file__).with_name("audit_sixs_final_project_integrity_generalization.py")), "exec"),
        namespace, namespace,
    )


if __name__ == "__main__":
    main()
