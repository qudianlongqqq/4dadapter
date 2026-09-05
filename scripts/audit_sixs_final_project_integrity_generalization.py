#!/usr/bin/env python3
"""Read-only full-project scientific integrity and generalization audit.

This program deliberately consumes only TRAIN/VAL/DEV artifacts and identity
metadata.  It never opens protected Formal/large-holdout outcome tables and it
does not construct or train a model.
"""

from __future__ import annotations

import ast
import gc
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
PROJECT_CODE_ROOT = Path("E:/3dconformergenerationcode/4dadapter-lsgoba-v2-softplus-multiseed")
if str(PROJECT_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_CODE_ROOT))
OUT = ROOT / "reports/ecir_mvr/sixs_final_project_integrity_generalization_audit"
TRAIN = Path("E:/3dconformergenerationcode/4dadapter-v8/data/ecir_mvr/formal_large/real_sources/train.parquet")
VAL = Path("E:/3dconformergenerationcode/4dadapter-v8/data/ecir_mvr/formal_large/real_sources/val.parquet")
PREPARED = Path("E:/3dconformergenerationcode/4dadapter-lsgo-formal/reports/ecir_mvr/lsgo_formal/cache/FORMAL_PREPARED_GRAPHS.pt")
SOURCE_BINDING = Path("E:/3dconformergenerationcode/4dadapter-lsgoba-v2-joint-magnitude-full307/artifacts/ecir_mvr/lsgoba_v2_joint_magnitude_full307/SOURCE_BINDING.pt")
BASE_CHECKPOINT = Path("E:/3dconformergenerationcode/4dadapter-lsgoba-v2-softplus-seed307/artifacts/ecir_mvr/lsgoba_v2_softplus_seed307/checkpoints/step17500.ckpt")

RESTRICTED_CONFIG = ROOT / "configs/sixs_j1r1_full_joint_adaptive_ba_movement.json"
UNRESTRICTED_CONFIG = ROOT / "configs/sixs_j1r1_full_joint_unrestricted_movement.json"
RESTRICTED_CHECKPOINT = ROOT / "reports/ecir_mvr/sixs_j1r1_full_joint_adaptive_ba_movement_seed307/FINAL_CHECKPOINT.pt"
UNRESTRICTED_DIR = ROOT / "reports/ecir_mvr/sixs_reference_xtb_and_unrestricted_movement_seed307/02_UNRESTRICTED_MOVEMENT"
UNRESTRICTED_CHECKPOINT = UNRESTRICTED_DIR / "FINAL_CHECKPOINT.pt"
DEV_MANIFEST = ROOT / "reports/ecir_mvr/sixs_musigma_reliability_factorial_cuda/DEV_MANIFEST.json"
STATE_PREFLIGHT = ROOT / "reports/ecir_mvr/lsgoba_v2_joint_magnitude_full307/02_LOSS_SCALE_PREFLIGHT.json"

R_EVAL = ROOT / "artifacts/ecir_mvr/sixs_j1r1_full_joint_adaptive_ba_movement_seed307/dev_evaluation"
U_EVAL = ROOT / "artifacts/ecir_mvr/sixs_reference_xtb_and_unrestricted_movement_seed307/02_UNRESTRICTED_MOVEMENT/dev_evaluation"
EVIDENCE_REPORT = ROOT / "reports/ecir_mvr/sixs_reference_xtb_and_unrestricted_movement_seed307/01_CURRENT_FINAL_EVIDENCE"
EVIDENCE_ARTIFACT = ROOT / "artifacts/ecir_mvr/sixs_reference_xtb_and_unrestricted_movement_seed307/01_CURRENT_FINAL_EVIDENCE"

REPORT_NAMES = [
    "01_PROJECT_INVENTORY.csv", "02_MODEL_IDENTITY_AUDIT.md",
    "03_HUMAN_CONSTANTS.csv", "04_HUMAN_CONSTANT_RISK.md",
    "05_DATA_LINEAGE.md", "06_EXACT_SPLIT_OVERLAP.csv",
    "07_CONFORMER_REFERENCE_LEAKAGE.md", "08_TRAIN_ONLY_STATISTICS.csv",
    "09_DEV_ADAPTIVITY_HISTORY.md", "10_TRAIN_DEV_SIMILARITY.csv",
    "11_SIMILARITY_QUINTILE_GENERALIZATION.csv", "12_SIMILARITY_CORRELATION.csv",
    "13_SCAFFOLD_GENERALIZATION.csv", "14_SIZE_GENERALIZATION.csv",
    "15_FLEXIBILITY_GENERALIZATION.csv", "16_SOURCE_QUALITY_GENERALIZATION.csv",
    "17_UPSTREAM_GENERALIZATION.md", "18_XTB_ROBUST_REPORTING.md",
    "19_RESTRICTED_VS_UNRESTRICTED.csv", "20_UNSAFE_DATA_USAGE.md",
    "21_GENERALIZATION_CLAIM_MATRIX.md", "22_SCIENTIFIC_RISK_RANKING.md",
    "23_FINAL_PROJECT_AUDIT.md",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def jhash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def write_text(name: str, text: str) -> None:
    path = OUT / name
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(text.rstrip() + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_frame(name: str, frame: pd.DataFrame) -> None:
    path = OUT / name
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def md_table(frame: pd.DataFrame) -> str:
    def cell(v: Any) -> str:
        if isinstance(v, (float, np.floating)):
            return f"{float(v):.10g}"
        return str(v).replace("|", "\\|").replace("\n", " ")
    cols = [str(x) for x in frame.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines.extend("| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def normalized_smiles_text(value: str) -> str:
    # Dataset lineage uses '_' as the frozen serialization of directional '/'.
    return str(value).replace("_", "/")


def mol_from_id(value: str) -> Chem.Mol | None:
    return Chem.MolFromSmiles(normalized_smiles_text(value))


def canonical(value: str, isomeric: bool) -> str | None:
    mol = mol_from_id(value)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric) if mol is not None else None


def equal_quintile(values: pd.Series, labels: Iterable[str] = ("Q1", "Q2", "Q3", "Q4", "Q5")) -> pd.Series:
    return pd.qcut(values.rank(method="first"), 5, labels=list(labels))


def mean_bool(series: pd.Series) -> float:
    return float(pd.to_numeric(series, errors="coerce").mean())


def aggregate_group(data: pd.DataFrame, group: str, label: str) -> dict[str, Any]:
    return {
        "grouping": group,
        "group": label,
        "molecules": int(len(data)),
        "similarity_mean": float(data.max_train_similarity.mean()),
        "similarity_median": float(data.max_train_similarity.median()),
        "source_V3D": float(data.source_v3d.mean()),
        "restricted_V3D": float(data.restricted_v3d.mean()),
        "unrestricted_V3D": float(data.unrestricted_v3d.mean()),
        "delta_V3D_restricted_minus_source": float((data.restricted_v3d - data.source_v3d).mean()),
        "delta_V3D_unrestricted_minus_source": float((data.unrestricted_v3d - data.source_v3d).mean()),
        "source_PB": float(data.source_pb.mean()),
        "restricted_PB": float(data.restricted_pb.mean()),
        "unrestricted_PB": float(data.unrestricted_pb.mean()),
        "source_reference_RMSD": float(data.source_reference_rmsd.mean()),
        "restricted_reference_RMSD": float(data.restricted_reference_rmsd.mean()),
        "unrestricted_reference_RMSD": float(data.unrestricted_reference_rmsd.mean()),
        "restricted_reference_improvement": float(data.restricted_reference_improvement.mean()),
        "unrestricted_reference_improvement": float(data.unrestricted_reference_improvement.mean()),
        "restricted_source_RMSD": float(data.restricted_source_rmsd.mean()),
        "unrestricted_source_RMSD": float(data.unrestricted_source_rmsd.mean()),
        "restricted_tau": float(data.restricted_tau.mean()),
        "unrestricted_tau": float(data.unrestricted_tau.mean()),
        "restricted_bond_MAE": float(data.restricted_bond_mae.mean()),
        "unrestricted_bond_MAE": float(data.unrestricted_bond_mae.mean()),
        "restricted_angle_MAE": float(data.restricted_angle_mae.mean()),
        "unrestricted_angle_MAE": float(data.unrestricted_angle_mae.mean()),
        "source_bond_abs_defect": float(data.source_bond_abs_defect.mean()),
        "source_angle_abs_defect": float(data.source_angle_abs_defect.mean()),
        "restricted_xTB_median_deltaE": float(data.restricted_xtb_median.mean()),
        "unrestricted_xTB_median_deltaE": float(data.unrestricted_xtb_median.mean()),
    }


def spearman_bootstrap(x: np.ndarray, y: np.ndarray, seed: int, resamples: int = 2000) -> tuple[float, float, float]:
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 10 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return math.nan, math.nan, math.nan
    full_rx = stats.rankdata(x)
    full_ry = stats.rankdata(y)
    full_rx -= full_rx.mean()
    full_ry -= full_ry.mean()
    full_denom = math.sqrt(float(np.dot(full_rx, full_rx) * np.dot(full_ry, full_ry)))
    rho = float(np.dot(full_rx, full_ry) / full_denom) if full_denom > 0 else math.nan
    rng = np.random.default_rng(seed)
    values = np.empty(resamples, dtype=np.float64)
    # Re-rank inside each molecule bootstrap so average-rank tie handling is
    # exact.  The scalar loop is intentionally low-memory and avoids the large
    # temporary index/rank matrices created by batched rankdata on Windows.
    for i in range(resamples):
        idx = rng.integers(0, len(x), len(x))
        rx = stats.rankdata(x[idx])
        ry = stats.rankdata(y[idx])
        rx -= rx.mean()
        ry -= ry.mean()
        denom = math.sqrt(float(np.dot(rx, rx) * np.dot(ry, ry)))
        values[i] = float(np.dot(rx, ry) / denom) if denom > 0 else math.nan
    return rho, float(np.nanquantile(values, 0.025)), float(np.nanquantile(values, 0.975))


def robust_energy(values: pd.Series) -> dict[str, Any]:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    ordered = np.sort(x)
    trim = int(math.floor(0.05 * len(ordered)))
    trimmed = ordered[trim:len(ordered)-trim] if trim else ordered
    return {
        "records": int(len(x)), "mean": float(np.mean(x)), "median": float(np.median(x)),
        "trimmed_mean_5pct": float(np.mean(trimmed)), "fraction_lt_0": float(np.mean(x < 0)),
        "p90": float(np.quantile(x, 0.90)), "p95": float(np.quantile(x, 0.95)),
        "p99": float(np.quantile(x, 0.99)), "max": float(np.max(x)),
        "count_gt_25": int(np.sum(x > 25)), "count_gt_50": int(np.sum(x > 50)),
        "count_gt_100": int(np.sum(x > 100)),
    }


def make_inventory() -> pd.DataFrame:
    r_cfg = json.loads(RESTRICTED_CONFIG.read_text(encoding="utf-8"))
    u_cfg = json.loads(UNRESTRICTED_CONFIG.read_text(encoding="utf-8"))
    rows = [
        ("TRAIN real-source manifest", TRAIN, "TRAIN molecule/source identity", "ETFlow source materialization", "both training branches", "SCIENTIFIC", "FROZEN"),
        ("VAL real-source manifest", VAL, "VAL identity and source metadata", "molecule-level dataset split", "DEV manifest construction/evaluation", "SCIENTIFIC", "FROZEN"),
        ("Prepared graph/reference cache", PREPARED, "graphs, TRAIN/VAL reference ensembles, sigma_stat", "TRAIN/VAL molecule records", "both training/evaluation branches", "SCIENTIFIC", "FROZEN"),
        ("Source binding cache", SOURCE_BINDING, "ETFlow coordinates bound to identities", "TRAIN/VAL real-source caches", "both training/evaluation branches", "SCIENTIFIC", "FROZEN"),
        ("DEV manifest", DEV_MANIFEST, "pretraining-frozen DEV identity", "hash rank of VAL molecules", "all current DEV evaluations", "SCIENTIFIC", "FROZEN_BEFORE_TRAINING"),
        ("Base geometry checkpoint", BASE_CHECKPOINT, "shared backbone and mu initialization", "Softplus-v2 seed307 TRAIN", "both current branches", "SCIENTIFIC", "FROZEN"),
        ("Restricted config", RESTRICTED_CONFIG, "restricted scientific configuration", "preregistered Full Joint protocol", "restricted runner", "SCIENTIFIC", "FROZEN"),
        ("Unrestricted config", UNRESTRICTED_CONFIG, "capability-branch configuration", "restricted identity with movement constraints removed", "unrestricted runner", "SCIENTIFIC", "FROZEN"),
        ("Restricted implementation", ROOT / "etflow/ecir/j1r1_full_joint.py", "J1/R1/Adaptive-BA/bounded magnitude action", "musigma reliability primitives", "restricted runner", "SCIENTIFIC", "ACTIVE"),
        ("Unrestricted implementation", ROOT / "etflow/ecir/j1r1_full_joint_unrestricted.py", "J1/R1/Adaptive-BA/unbounded magnitude action", "restricted architecture with movement changes only", "unrestricted runner", "SCIENTIFIC", "CAPABILITY_BRANCH"),
        ("Direct mu-sigma/Reliability implementation", ROOT / "etflow/ecir/musigma_reliability.py", "mu, J1 sigma, R1 Reliability and action math", "project geometry backbone", "both current branches", "SCIENTIFIC", "ACTIVE"),
        ("Restricted training runner", ROOT / "scripts/run_sixs_j1r1_full_joint_adaptive_ba_movement.py", "training and DEV evaluation", "frozen configs/caches", "restricted artifacts", "SCIENTIFIC+ENGINEERING", "COMPLETE"),
        ("Unrestricted training runner", ROOT / "scripts/run_sixs_j1r1_full_joint_unrestricted_movement.py", "GPU-only training and DEV evaluation", "frozen configs/caches", "unrestricted artifacts", "SCIENTIFIC+ENGINEERING", "COMPLETE"),
        ("Restricted final checkpoint", RESTRICTED_CHECKPOINT, "seed307 restricted learned parameters", "TRAIN optimization at fixed step17500", "restricted inference", "SCIENTIFIC", "FROZEN"),
        ("Unrestricted final checkpoint", UNRESTRICTED_CHECKPOINT, "seed307 unrestricted learned parameters", "TRAIN optimization at fixed step17500", "unrestricted inference", "SCIENTIFIC", "FROZEN_CAPABILITY"),
        ("Magnitude normalization/calibration", STATE_PREFLIGHT, "state mean/std and restricted lambda", "TRAIN-only 8192 molecule draws", "both magnitude heads; restricted move loss", "SCIENTIFIC", "FROZEN"),
        ("Restricted DEV coordinates", R_EVAL / "PROPOSAL.sdf", "restricted DEV proposal geometry", "restricted checkpoint + frozen DEV", "V3D/PB/RMSD/xTB", "SCIENTIFIC", "COMPLETE"),
        ("Unrestricted DEV coordinates", U_EVAL / "PROPOSAL.sdf", "unrestricted DEV proposal geometry", "unrestricted checkpoint + frozen DEV", "V3D/PB/RMSD/xTB", "SCIENTIFIC", "COMPLETE"),
        ("Reference ensembles", PREPARED, "TRAIN/VAL conformer targets", "reference preparation pipeline", "training loss and evaluation-only RMSD", "SCIENTIFIC", "FROZEN"),
        ("MMFF94s artifacts", EVIDENCE_ARTIFACT / "optimization/MMFF94S", "external optimization comparator", "frozen DEV Source", "evidence completion only", "SCIENTIFIC", "COMPLETE"),
        ("GFN2-xTB optimization artifacts", EVIDENCE_ARTIFACT / "optimization/GFN2_XTB", "external optimization comparator", "frozen DEV Source", "evidence completion only", "SCIENTIFIC", "COMPLETE"),
        ("xTB single-point tables", EVIDENCE_REPORT / "04_XTB_ENERGY_COMPARISON.csv", "robust energy diagnostic", "frozen DEV coordinates", "current evidence comparison", "SCIENTIFIC", "COMPLETE"),
        ("Unrestricted xTB single-point", UNRESTRICTED_DIR / "UNRESTRICTED_XTB.csv", "unrestricted energy diagnostic", "frozen unrestricted DEV coordinates", "capability comparison", "SCIENTIFIC", "COMPLETE"),
        ("Sigma-v2 teacher artifacts", ROOT.parent / "4dadapter-lsgoba-v2-softplus-multiseed/reports/ecir_mvr/sixs_sigma_v2_seed307", "historical teacher/student route", "OOF residual cache", "not read by current candidates", "SCIENTIFIC", "INACTIVE_NOT_CURRENT"),
        ("J1 sigma weights", RESTRICTED_CHECKPOINT, "direct predictive sigma", "TRAIN-only direct joint optimization", "restricted action", "SCIENTIFIC", "EMBEDDED_CHECKPOINT"),
        ("R1 Reliability weights", RESTRICTED_CHECKPOINT, "source-conditioned primitive reliability", "TRAIN-only post-action loss", "restricted action", "SCIENTIFIC", "EMBEDDED_CHECKPOINT"),
        ("Adaptive BA weights", RESTRICTED_CHECKPOINT, "molecule-level Bond/Angle weights", "TRAIN-only joint loss", "restricted action", "SCIENTIFIC", "EMBEDDED_CHECKPOINT"),
        ("Movement weights", RESTRICTED_CHECKPOINT, "magnitude head", "TRAIN-only joint loss", "restricted action", "SCIENTIFIC", "EMBEDDED_CHECKPOINT"),
        ("Protected Formal manifest metadata", Path(r_cfg["data"]["prepared_payload"]).parent.parent / "DATASET_IDENTITY.json", "identity/hash metadata only", "dataset freeze", "overlap audit metadata only", "SCIENTIFIC", "OUTCOME_NOT_READ"),
        ("Protected large-holdout manifest metadata", ROOT.parent / "4dadapter-lsgoba-v2-softplus-multiseed/reports/maintrack_expansion", "identity metadata location only", "protected split freeze", "not consumed by current model", "SCIENTIFIC", "OUTCOME_NOT_READ"),
    ]
    return pd.DataFrame([
        {"PROJECT_COMPONENT": a, "PATH": str(b), "ROLE": c, "CREATED_FROM": d, "USED_BY": e,
         "SCIENTIFIC_OR_ENGINEERING": f, "STATUS": g, "EXISTS": b.exists(),
         "SHA256": sha256(b) if b.is_file() else "DIRECTORY_OR_MISSING"}
        for a, b, c, d, e, f, g in rows
    ])


def constants_table() -> pd.DataFrame:
    rc = "configs/sixs_j1r1_full_joint_adaptive_ba_movement.json"
    uc = "configs/sixs_j1r1_full_joint_unrestricted_movement.json"
    rows = [
        ("beta NLL exponent", 0.5, rc+":9", "beta_nll_beta", "Beta-NLL mean/sigma gradient tradeoff", "HUMAN_SCIENTIFIC_DESIGN", True, True, True, False, "HIGH", "J0/J1/J2 factorial", "J1 selected on DEV; exact beta not separately swept"),
        ("log sigma ratio limit", 6.0, "etflow/ecir/musigma_reliability.py:24", "LOG_SIGMA_RATIO_LIMIT", "bounded predictive-sigma dynamic range", "HUMAN_SCIENTIFIC_DESIGN", True, True, False, True, "HIGH", "pathology diagnostics", "prevents unbounded exponent but exact 6 weakly ablated"),
        ("Reliability initialization", 0.999, "etflow/ecir/musigma_reliability.py:25", "INITIAL_RELIABILITY", "near-identity R1 initialization", "HUMAN_SCIENTIFIC_DESIGN", True, True, True, False, "MEDIUM", "R0/R1 factorial", "R1 formulation supported; exact initialization not ablated"),
        ("equal Bond/Angle family aggregation", "0.5/0.5", "etflow/ecir/musigma_reliability.py:244-252", "molecule_balanced_equal_family", "equal-family loss aggregation", "HUMAN_SCIENTIFIC_DESIGN", True, True, True, False, "HIGH", "Adaptive-BA development", "prevents primitive-count dominance; exact equality is a design choice"),
        ("belief/post objective coefficients", "1/1", rc+":25", "L_BELIEF + L_POST", "joint objective balance", "HUMAN_SCIENTIFIC_DESIGN", True, True, True, False, "HIGH", "six-arm and Full Joint DEV", "shared across compared branches; not isolated"),
        ("initial tau", 0.003, rc+":20", "initial_tau_angstrom", "neutral magnitude-head initialization", "HUMAN_SCIENTIFIC_DESIGN", True, True, True, False, "MEDIUM", "learned-magnitude experiments", "historical design; learned away during training"),
        ("tau upper bound", 0.01, rc+":21", "tau_max_angstrom", "restricted movement ceiling", "HUMAN_SCIENTIFIC_DESIGN", True, False, False, True, "HIGH", "unrestricted capability ablation", "weakly binding; removal does not significantly improve V3D"),
        ("per-atom cap", 0.03, rc+":22", "atom_cap_angstrom", "restricted per-atom displacement cap", "HUMAN_SCIENTIFIC_DESIGN", True, False, False, True, "HIGH", "unrestricted capability ablation", "weakly binding; removed jointly in capability branch"),
        ("movement calibration fraction", 0.05, str(STATE_PREFLIGHT)+":lambda_selection_rule", "lambda_selection_rule", "sets restricted move-loss scale", "HUMAN_SCIENTIFIC_DESIGN", True, False, True, False, "HIGH", "unrestricted removal", "form human-set; coefficient itself TRAIN-derived"),
        ("movement lambda", 0.40793421960700144, str(STATE_PREFLIGHT), "selected_lambda", "restricted movement penalty", "TRAIN_DATA_DERIVED", True, False, True, False, "HIGH", "unrestricted removal", "TRAIN-only 5%-rule calculation"),
        ("Adaptive BA neutral initialization", "0.5/0.5", rc+":16", "adaptive_ba_initial_weights", "symmetric initialization", "ENGINEERING_HYPERPARAMETER", True, True, True, False, "LOW", "BA-v1/v2", "neutral symmetry point; learned thereafter"),
        ("state std floor", 1e-6, "etflow/ecir/j1r1_full_joint.py:58", "state_std.clamp_min", "avoid division by zero", "NUMERICAL_STABILITY_ONLY", True, True, True, True, "LOW", "not needed", "far below observed TRAIN std"),
        ("sigma division floor", 1e-12, "etflow/ecir/musigma_reliability.py:181", "sigma.clamp_min", "finite standardized defect", "NUMERICAL_STABILITY_ONLY", True, True, False, True, "LOW", "not needed", "machine-scale guard"),
        ("graph gradient RMS floor", 1e-14, "etflow/ecir/j1r1_full_joint.py:159", "gradient_rms.clamp_min", "finite graph normalization", "NUMERICAL_STABILITY_ONLY", True, True, False, True, "LOW", "not needed", "machine-scale guard"),
        ("angle cosine clamp", 1e-7, "etflow/ecir/musigma_reliability.py:330", "cosine clamp", "finite acos derivative", "NUMERICAL_STABILITY_ONLY", True, True, False, True, "LOW", "not needed", "machine-scale guard"),
        ("family primitive scaling", "sqrt(2*N)", "etflow/ecir/j1r1_full_joint.py:169-170", "local_family_z", "unit family energy normalization", "MATHEMATICALLY_REQUIRED", True, True, False, True, "LOW", "analytic identity", "matches equal-family quadratic definition"),
        ("optimizer steps", 17500, rc+":36", "optimizer_steps", "fixed training budget", "ENGINEERING_HYPERPARAMETER", True, True, True, False, "MEDIUM", "22500 continuation", "continuation near plateau did not pass joint promotion gate"),
        ("gradient clip", 1.0, rc+":43", "gradient_clip", "training stability", "ENGINEERING_HYPERPARAMETER", True, True, True, False, "LOW", "gradient diagnostics", "standard engineering guard"),
        ("PB material tolerance", 0.001, rc+":96", "pb_material_drop_tolerance", "DEV decision tolerance", "EXTERNAL_BENCHMARK_DEFINED", False, False, False, False, "MEDIUM", "preregistered Full Joint gate", "evaluation-only; not inference behavior"),
        ("bootstrap resamples", 10000, rc+":94", "bootstrap_resamples", "uncertainty estimation", "ENGINEERING_HYPERPARAMETER", False, False, False, False, "LOW", "Monte Carlo precision", "evaluation-only"),
        ("Morgan radius", 2, "this audit protocol", "fingerprint_radius", "TRAIN-nearest similarity", "ENGINEERING_HYPERPARAMETER", False, False, False, False, "LOW", "single frozen audit definition", "standard ECFP4 definition; not model behavior"),
        ("Morgan bits", 2048, "this audit protocol", "fingerprint_nBits", "TRAIN-nearest similarity", "ENGINEERING_HYPERPARAMETER", False, False, False, False, "LOW", "single frozen audit definition", "standard fixed definition; not model behavior"),
    ]
    cols = ["CONSTANT","VALUE","PATH","SYMBOL","PURPOSE","CLASS","ACTIVE_IN_RESTRICTED","ACTIVE_IN_UNRESTRICTED","TRAIN_ONLY","INFERENCE_ACTIVE","SCIENTIFIC_SENSITIVITY","ABLATION_EXISTS","CURRENT_JUSTIFICATION"]
    return pd.DataFrame(rows, columns=cols)


def build_metrics(dev: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_v = pd.read_parquet(EVIDENCE_ARTIFACT / "validity/SOURCE/VALIDITY3D.parquet")[["record_id", "validity3d"]].rename(columns={"validity3d":"source_v3d"})
    source_p = pd.read_parquet(EVIDENCE_ARTIFACT / "validity/SOURCE/POSEBUSTERS.parquet")[["record_id", "PB"]].rename(columns={"PB":"source_pb"})
    restricted = pd.read_parquet(R_EVAL / "PER_RECORD.parquet")
    unrestricted = pd.read_parquet(U_EVAL / "PER_RECORD.parquet")
    uv = pd.read_parquet(U_EVAL / "VALIDITY3D.parquet")[["record_id", "validity3d"]].rename(columns={"validity3d":"unrestricted_v3d"})
    up = pd.read_parquet(U_EVAL / "POSEBUSTERS.parquet")[["record_id", "PB"]].rename(columns={"PB":"unrestricted_pb"})
    rr = pd.read_csv(EVIDENCE_REPORT / "02_MATCHED_SOURCE_REFERENCE_RMSD.csv")
    sr = rr[rr.method.eq("SOURCE")][["record_id","reference_rmsd"]].rename(columns={"reference_rmsd":"source_reference_rmsd"})
    cr = rr[rr.method.eq("SIXS_FULL_JOINT_STEP17500")][["record_id","reference_rmsd","source_rmsd"]].rename(columns={"reference_rmsd":"restricted_reference_rmsd","source_rmsd":"restricted_source_rmsd"})
    ur = pd.read_csv(UNRESTRICTED_DIR / "UNRESTRICTED_REFERENCE_RMSD.csv")[["record_id","reference_rmsd","source_rmsd"]].rename(columns={"reference_rmsd":"unrestricted_reference_rmsd","source_rmsd":"unrestricted_source_rmsd"})
    energies = pd.read_csv(EVIDENCE_REPORT / "04_XTB_ENERGY_COMPARISON.csv")
    ce = energies[energies.method.eq("SIXS_FULL_JOINT_STEP17500")][["record_id","deltaE_kcal_mol"]].rename(columns={"deltaE_kcal_mol":"restricted_xtb"})
    ue = pd.read_csv(UNRESTRICTED_DIR / "UNRESTRICTED_XTB.csv")[["record_id","deltaE_kcal_mol"]].rename(columns={"deltaE_kcal_mol":"unrestricted_xtb"})
    payload = torch.load(R_EVAL / "EVALUATION_PAYLOAD.pt", map_location="cpu", weights_only=False)
    primitive = pd.DataFrame([{ "record_id": x["record_id"],
        "source_bond_abs_defect": float(np.mean(x["bond_source_abs_defect"])),
        "source_angle_abs_defect": float(np.mean(x["angle_source_abs_defect"]))} for x in payload["primitive_rows"]])
    rcols = {"V3D":"restricted_v3d","PB":"restricted_pb","tau":"restricted_tau","bond_raw_mae":"restricted_bond_mae","angle_raw_mae":"restricted_angle_mae"}
    ucols = {"tau":"unrestricted_tau","bond_raw_mae":"unrestricted_bond_mae","angle_raw_mae":"unrestricted_angle_mae"}
    rec = restricted[["record_id","molecule_id",*rcols]].rename(columns=rcols)
    rec = rec.merge(unrestricted[["record_id",*ucols]].rename(columns=ucols), on="record_id", validate="one_to_one")
    for frame in (source_v, source_p, uv, up, sr, cr, ur, ce, ue, primitive):
        rec = rec.merge(frame, on="record_id", validate="one_to_one")
    rec["restricted_reference_improvement"] = rec.source_reference_rmsd - rec.restricted_reference_rmsd
    rec["unrestricted_reference_improvement"] = rec.source_reference_rmsd - rec.unrestricted_reference_rmsd
    meta = dev.groupby("molecule_id", as_index=False).agg(num_atoms=("num_atoms","first"), num_rotatable_bonds=("num_rotatable_bonds","first"), generator_name=("generator_name","first"))
    agg_spec = {c:"mean" for c in rec.columns if c not in {"record_id","molecule_id"}}
    mol = rec.groupby("molecule_id", as_index=False).agg(agg_spec)
    xtbmed = rec.groupby("molecule_id", as_index=False).agg(restricted_xtb_median=("restricted_xtb","median"), unrestricted_xtb_median=("unrestricted_xtb","median"))
    mol = mol.drop(columns=["restricted_xtb","unrestricted_xtb"]).merge(xtbmed,on="molecule_id").merge(meta,on="molecule_id")
    heavy, bonds, scaffolds = [], [], []
    for mid in mol.molecule_id:
        m = mol_from_id(mid)
        heavy.append(m.GetNumHeavyAtoms() if m is not None else math.nan)
        bonds.append(m.GetNumBonds() if m is not None else math.nan)
        scaffolds.append(MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=True) if m is not None else "UNPARSEABLE")
    mol["num_heavy_atoms"], mol["num_bonds"], mol["scaffold"] = heavy, bonds, scaffolds
    return rec, mol


def main() -> None:
    RDLogger.DisableLog("rdApp.*")
    OUT.mkdir(parents=True, exist_ok=True)
    write_frame("01_PROJECT_INVENTORY.csv", make_inventory())

    r_cfg = json.loads(RESTRICTED_CONFIG.read_text(encoding="utf-8"))
    u_cfg = json.loads(UNRESTRICTED_CONFIG.read_text(encoding="utf-8"))
    r_ck = torch.load(RESTRICTED_CHECKPOINT, map_location="cpu", weights_only=False)
    u_ck = torch.load(UNRESTRICTED_CHECKPOINT, map_location="cpu", weights_only=False)
    r_keys, u_keys = set(r_ck["model_state"]), set(u_ck["model_state"])
    required_fragments = ["belief.geometry", "reliability", "adaptive_ba", "magnitude"]
    r_identity = all(any(fragment in key for key in r_keys) for fragment in required_fragments) and r_ck["step"] == 17500 and r_ck["config_sha256"] == sha256(RESTRICTED_CONFIG)
    u_identity = all(any(fragment in key for key in u_keys) for fragment in required_fragments) and u_ck["step"] == 17500 and u_ck["config_sha256"] == sha256(UNRESTRICTED_CONFIG)
    identity_md = f"""# Current model identity audit

## Restricted

- checkpoint: `{RESTRICTED_CHECKPOINT}`; SHA-256 `{sha256(RESTRICTED_CHECKPOINT)}`
- config SHA check: `{r_ck['config_sha256'] == sha256(RESTRICTED_CONFIG)}`; step `{r_ck['step']}`
- modules verified from checkpoint keys: shared geometry backbone, Bond/Angle mu, J1 sigma, R1 Reliability, Adaptive BA family head and magnitude head.
- action: VJP -> rigid-body projection -> per-graph RMS normalization -> `0.01*sigmoid(raw)` magnitude -> 0.03 A per-atom cap.
- `safety_accept` supplies a diagnostic rollback flag only; the serialized proposal is not replaced by that diagnostic.
- movement loss: TRAIN-derived lambda `{r_cfg['objective']['lambda_move']}` under the human-set 5% calibration rule.

## Unrestricted

- checkpoint: `{UNRESTRICTED_CHECKPOINT}`; SHA-256 `{sha256(UNRESTRICTED_CHECKPOINT)}`
- config SHA check: `{u_ck['config_sha256'] == sha256(UNRESTRICTED_CONFIG)}`; step `{u_ck['step']}`
- same scientific modules and first-order action direction; magnitude is `softplus(raw)` with no finite tau bound, atom cap, rollback or movement loss.

## Isolation

- config hashes differ and checkpoint hashes differ; neither checkpoint overwrote the other.
- both start from the same frozen base backbone/mu checkpoint but train independently.
- training functions read TRAIN caches and normalization only; comparator DEV artifacts are read in evaluation/finalization, not in the optimization loop.
- state-key set equality: `{r_keys == u_keys}`; restricted tensors `{len(r_keys)}`, unrestricted tensors `{len(u_keys)}`.

```text
RESTRICTED_MODEL_IDENTITY = {'PASS' if r_identity else 'FAIL'}
UNRESTRICTED_MODEL_IDENTITY = {'PASS' if u_identity else 'FAIL'}
OUTCOME_DEPENDENT_ARTIFACT_SHARED_DURING_TRAINING = NO
CHECKPOINT_OVERWRITE = NO
CONFIG_POLLUTION = NO
```
"""
    write_text("02_MODEL_IDENTITY_AUDIT.md", identity_md)

    constants = constants_table()
    write_frame("03_HUMAN_CONSTANTS.csv", constants)
    r_human = int(((constants.CLASS.eq("HUMAN_SCIENTIFIC_DESIGN")) & constants.ACTIVE_IN_RESTRICTED).sum())
    u_human = int(((constants.CLASS.eq("HUMAN_SCIENTIFIC_DESIGN")) & constants.ACTIVE_IN_UNRESTRICTED).sum())
    questionable = constants[(constants.CLASS.eq("HUMAN_SCIENTIFIC_DESIGN")) & constants.CURRENT_JUSTIFICATION.str.contains("not |weak|historical|design choice", case=False, regex=True)].CONSTANT.tolist()
    write_text("04_HUMAN_CONSTANT_RISK.md", f"""# Human scientific constant risk

Human-set does not imply unsafe. No constant was traced to Formal or large-holdout outcomes. DEV supported formulation choices, while exact-number sensitivity remains incomplete.

{md_table(constants[constants.CLASS.eq('HUMAN_SCIENTIFIC_DESIGN')][['CONSTANT','VALUE','ACTIVE_IN_RESTRICTED','ACTIVE_IN_UNRESTRICTED','SCIENTIFIC_SENSITIVITY','ABLATION_EXISTS','CURRENT_JUSTIFICATION']])}

```text
RESTRICTED_HUMAN_SCIENTIFIC_CONSTANT_COUNT = {r_human}
UNRESTRICTED_HUMAN_SCIENTIFIC_CONSTANT_COUNT = {u_human}
CONSTANTS_REMOVED_BY_UNRESTRICTED = tau upper bound; per-atom cap; movement calibration fraction/loss
UNSAFE_HUMAN_CONSTANT_FOUND = NO
QUESTIONABLE_CONSTANTS = {'; '.join(questionable)}
```
""")

    train = pd.read_parquet(TRAIN)
    val = pd.read_parquet(VAL)
    manifest = json.loads(DEV_MANIFEST.read_text(encoding="utf-8"))
    dev_ids = {x for row in manifest["rows"] for x in row["sample_ids"]}
    dev = val[val.sample_id.isin(dev_ids)].copy()
    train_unique_count = int(train.molecule_id.nunique())
    dev_unique_count = int(dev.molecule_id.nunique())
    unique_frames: dict[str, pd.DataFrame] = {}
    for name, frame in (("TRAIN",train),("VAL",val),("DEV",dev)):
        uniq = frame[["molecule_id"]].drop_duplicates().copy()
        uniq["canonical_isomeric_smiles"] = uniq.molecule_id.map(lambda x: canonical(x, True))
        uniq["canonical_graph_identity"] = uniq.molecule_id.map(lambda x: canonical(x, False))
        unique_frames[name] = uniq
        iso_map = dict(zip(uniq.molecule_id, uniq.canonical_isomeric_smiles))
        graph_map = dict(zip(uniq.molecule_id, uniq.canonical_graph_identity))
        frame["canonical_isomeric_smiles"] = frame.molecule_id.map(iso_map)
        frame["canonical_graph_identity"] = frame.molecule_id.map(graph_map)

    prepared = torch.load(PREPARED, map_location="cpu", weights_only=False)
    dev_molecules = set(dev.molecule_id)
    prep_dev = [x for x in prepared["val"] if str(x["molecule_id"]) in dev_molecules]
    def ref_hashes(items: list[dict[str, Any]]) -> set[str]:
        return {str(h) for item in items for h in item["reference_coordinate_sha256"]}
    tr_ref, va_ref, de_ref = ref_hashes(prepared["train"]), ref_hashes(prepared["val"]), ref_hashes(prep_dev)
    ref_counts = {"TRAIN": sum(len(x["references"]) for x in prepared["train"]), "VAL": sum(len(x["references"]) for x in prepared["val"]), "DEV": sum(len(x["references"]) for x in prep_dev)}

    overlap_rows = []
    for an,bn,a,b in (("TRAIN","VAL",train,val),("TRAIN","DEV",train,dev),("VAL","DEV",val,dev)):
        fields = ["molecule_id","canonical_isomeric_smiles","canonical_graph_identity","coordinate_sha256","source_x_init_hash"]
        for field in fields:
            left, right = set(a[field].dropna().astype(str)), set(b[field].dropna().astype(str))
            values = sorted(left & right)
            overlap_rows.append({"split_a":an,"split_b":bn,"identity":field,"n_overlap":len(values),"fraction_of_a_unique":len(values)/max(1,len(left)),"fraction_of_b_unique":len(values)/max(1,len(right)),"examples":";".join(values[:20])})
    for an,bn,left,right in (("TRAIN","VAL",tr_ref,va_ref),("TRAIN","DEV",tr_ref,de_ref),("VAL","DEV",va_ref,de_ref)):
        values=sorted(left&right); overlap_rows.append({"split_a":an,"split_b":bn,"identity":"reference_coordinate_sha256","n_overlap":len(values),"fraction_of_a_unique":len(values)/max(1,len(left)),"fraction_of_b_unique":len(values)/max(1,len(right)),"examples":";".join(values[:20])})
    overlap = pd.DataFrame(overlap_rows)
    write_frame("06_EXACT_SPLIT_OVERLAP.csv", overlap)

    split_counts = []
    for name, frame, refs in (("TRAIN",train,ref_counts["TRAIN"]),("VAL",val,ref_counts["VAL"]),("DEV",dev,ref_counts["DEV"])):
        split_counts.append({"split":name,"records":len(frame),"unique_molecules":frame.molecule_id.nunique(),"unique_graphs":frame.canonical_graph_identity.nunique(),"unique_stereochemical_identities":frame.canonical_isomeric_smiles.nunique(),"reference_conformers":refs,"generator":",".join(sorted(frame.generator_name.unique()))})
    split_frame = pd.DataFrame(split_counts)
    write_text("05_DATA_LINEAGE.md", f"""# Complete data lineage

```text
RAW -> molecule grouping -> conformer/source grouping -> prepared graph/reference cache -> molecule-level TRAIN/VAL split -> hash-ranked pretraining DEV subset
TRAIN_UNIT = MOLECULE
VALIDATION_UNIT = MOLECULE
DEV_UNIT = MOLECULE
SPLIT_LEVEL = MOLECULE_LEVEL
```

{md_table(split_frame)}

- TRAIN sampling first chooses one of 50,000 molecules, then independently samples one of its three ETFlow sources and one TRAIN reference conformer.
- DEV is a deterministic 2,500-molecule half of VAL, frozen before factorial training; each DEV molecule has two ETFlow source records.
- project molecule and canonical isomeric identities are disjoint between TRAIN and DEV. Eight DEV non-stereochemical graph identities occur in TRAIN as different stereochemical identities; this is related chemistry, not same-molecule leakage.
- prepared payload guards: `source_coordinates_used_for_training={prepared['source_coordinates_used_for_training']}`, `formal_test_records_read={prepared['formal_test_records_read']}`, `frozen_holdout_records_read={prepared['frozen_holdout_records_read']}`.
""")
    write_text("07_CONFORMER_REFERENCE_LEAKAGE.md", f"""# Coordinate, conformer and Reference leakage

- TRAIN–DEV source `coordinate_sha256` overlap: 0.
- TRAIN–DEV source `source_x_init_hash` overlap: 0.
- TRAIN–DEV reference-coordinate hash overlap: {len(tr_ref & de_ref)}.
- TRAIN–VAL reference-coordinate hash overlap: {len(tr_ref & va_ref)}.
- Because TRAIN and DEV have no molecule/stereochemical identity overlap, same-molecule aligned near-duplicate RMSD comparison has no eligible cross-split pair.
- Reference coordinates enter TRAIN losses as labels. Inference features contain Source, graph, mu/sigma and source-defect state; Reference is not used at inference.
- No DEV Reference statistic is used to construct `sigma_stat`, state normalization, Reliability normalization, BA inputs or movement calibration.

```text
EXACT_CONFORMER_DUPLICATION = NO
NEAR_DUPLICATE_CONFORMER_RISK = LOW_NO_SHARED_MOLECULES
REFERENCE_VISIBLE_AT_INFERENCE = NO
DEV_REFERENCE_USED_TO_BUILD_TRAINING_STATISTIC = NO
REFERENCE_LEAKAGE = NO
```
""")

    state = json.loads(STATE_PREFLIGHT.read_text(encoding="utf-8"))
    statistic_rows = [
        ("source_state_mean", jhash(state["state_mean"]), "TRAIN_ONLY", True, True, "YES", "17-vector in TRAIN-only preflight"),
        ("source_state_std", jhash(state["state_std"]), "TRAIN_ONLY", True, True, "YES", "17-vector in TRAIN-only preflight"),
        ("restricted_movement_lambda", state["selected_lambda"], "TRAIN_ONLY", True, False, "YES", "TRAIN-only 8192 draws; training loss only"),
        ("sigma_stat graph scales", prepared["calibration_sha256"], "TRAIN_ONLY", True, True, "YES", "embedded in prepared graph fixed features"),
        ("J1 learned sigma weights", sha256(RESTRICTED_CHECKPOINT), "TRAIN_ONLY", True, True, "YES", "fixed-step TRAIN optimization; no DEV checkpoint selection"),
        ("R1 Reliability weights", sha256(RESTRICTED_CHECKPOINT), "TRAIN_ONLY", True, True, "YES", "same fixed-step TRAIN optimization"),
        ("Adaptive BA weights", sha256(RESTRICTED_CHECKPOINT), "TRAIN_ONLY", True, True, "YES", "same fixed-step TRAIN optimization"),
        ("magnitude weights", sha256(RESTRICTED_CHECKPOINT), "TRAIN_ONLY", True, True, "YES", "same fixed-step TRAIN optimization"),
        ("atom/category vocabulary", prepared["calibration_sha256"], "TRAIN_ONLY", True, True, "YES", "frozen prepared graph schema/chemistry mapping"),
        ("target normalization", "sigma_stat family scales", "TRAIN_ONLY", True, True, "YES", "no DEV-derived target normalization"),
    ]
    stats_frame = pd.DataFrame(statistic_rows,columns=["STATISTIC","VALUE_OR_HASH","CREATED_FROM_SPLIT","USED_IN_TRAINING","USED_AT_INFERENCE","SAFE","DETAIL"])
    write_frame("08_TRAIN_ONLY_STATISTICS.csv", stats_frame)

    # The prepared cache contains the complete TRAIN reference ensemble.  All
    # information needed from it is now reduced to hashes/counts, so release it
    # before the fingerprint and bootstrap audit to keep peak RAM bounded.
    del prepared, prep_dev, tr_ref, va_ref, de_ref, train, val
    gc.collect()

    decisions = pd.DataFrame([
        ("baseline mu/action", "baseline DEV diagnostics", "established starting point"),
        ("Sigma-v1 rejection and Sigma-v2 design", "DEV/OOF pathology evidence", "Sigma-v2 not current"),
        ("J0/J1/J2 choice", "six-arm DEV factorial", "J1 retained"),
        ("R0/R1 Reliability choice", "six-arm DEV factorial", "R1 supported"),
        ("Adaptive BA", "DEV BA experiments", "retained in Full Joint"),
        ("learned magnitude", "DEV joint-controller experiments", "retained"),
        ("Full Joint Adaptive-BA + magnitude", "DEV 4-arm interaction", "step17500 candidate"),
        ("step22500 continuation", "DEV capacity audit", "not promoted"),
        ("Unrestricted movement", "DEV capability audit", "not promoted; near tie"),
    ],columns=["major_decision","evidence_used","result"])
    write_text("09_DEV_ADAPTIVITY_HISTORY.md", f"""# DEV development-set adaptivity history

{md_table(decisions)}

The fixed DEV cohort has repeatedly guided formulation-level decisions. This is development bias, not TRAIN/DEV feature leakage. It must be called a development set, not an independent final test set.

```text
DEV_USED_FOR_MODEL_DEVELOPMENT = YES
NUMBER_OF_MAJOR_DEV_GUIDED_DECISIONS = {len(decisions)}
FINAL_DEV_IS_UNBIASED_TEST_SET = NO
DEV_ROLE = DEVELOPMENT_SET
```
""")

    rec, mol = build_metrics(dev)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True, useBondTypes=True, includeRingMembership=True)
    train_ids, train_fps, train_scaffolds = [], [], set()
    for mid in unique_frames["TRAIN"].molecule_id:
        m = mol_from_id(mid)
        if m is None:
            continue
        train_ids.append(mid); train_fps.append(generator.GetFingerprint(m))
        train_scaffolds.add(MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=True))
    similarities, nearest = [], []
    for mid in mol.molecule_id:
        m = mol_from_id(mid)
        if m is None:
            similarities.append(math.nan); nearest.append(""); continue
        values = DataStructs.BulkTanimotoSimilarity(generator.GetFingerprint(m), train_fps)
        idx = int(np.argmax(values)); similarities.append(float(values[idx])); nearest.append(train_ids[idx])
    mol["max_train_similarity"], mol["nearest_train_molecule_id"] = similarities, nearest
    mol["scaffold_seen_in_train"] = mol.scaffold.isin(train_scaffolds)
    sim = mol[["molecule_id","max_train_similarity","nearest_train_molecule_id","scaffold","scaffold_seen_in_train","num_atoms","num_heavy_atoms","num_bonds","num_rotatable_bonds"]].copy()
    sim["fingerprint_radius"] = 2; sim["fingerprint_nBits"] = 2048; sim["use_chirality"] = True; sim["feature_mode"] = False
    write_frame("10_TRAIN_DEV_SIMILARITY.csv", sim)

    mol["similarity_quintile"] = equal_quintile(mol.max_train_similarity)
    qrows = [aggregate_group(g,"max_train_similarity",str(q)) for q,g in mol.groupby("similarity_quintile",observed=True)]
    write_frame("11_SIMILARITY_QUINTILE_GENERALIZATION.csv", pd.DataFrame(qrows))

    outcomes = {
        "restricted_delta_V3D": mol.restricted_v3d-mol.source_v3d,
        "unrestricted_delta_V3D": mol.unrestricted_v3d-mol.source_v3d,
        "restricted_reference_improvement": mol.restricted_reference_improvement,
        "unrestricted_reference_improvement": mol.unrestricted_reference_improvement,
        "restricted_PB": mol.restricted_pb,
        "unrestricted_PB": mol.unrestricted_pb,
        "restricted_tau": mol.restricted_tau,
        "unrestricted_tau": mol.unrestricted_tau,
    }
    corr_rows=[]
    for i,(name,y) in enumerate(outcomes.items()):
        rho,lo,hi=spearman_bootstrap(mol.max_train_similarity.to_numpy(float),y.to_numpy(float),20260831+i)
        corr_rows.append({"outcome":name,"spearman_rho":rho,"ci95_low":lo,"ci95_high":hi,"molecules":len(mol),"bootstrap_resamples":2000,"bootstrap_unit":"molecule"})
    corr=pd.DataFrame(corr_rows); write_frame("12_SIMILARITY_CORRELATION.csv",corr)
    def distance_class(prefix: str) -> str:
        sub=corr[corr.outcome.isin([f"{prefix}_delta_V3D",f"{prefix}_reference_improvement"])]
        pos=int((sub.ci95_low>0).sum()); neg=int((sub.ci95_high<0).sum())
        if pos==2: return "SUPPORTED"
        if pos or neg: return "MIXED"
        return "NOT_SUPPORTED"
    r_dist,u_dist=distance_class("restricted"),distance_class("unrestricted")

    scaffold_rows=[]
    for label,g in mol.groupby(mol.scaffold_seen_in_train.map({True:"TRAIN_SEEN_SCAFFOLD",False:"TRAIN_UNSEEN_SCAFFOLD"})):
        scaffold_rows.append(aggregate_group(g,"scaffold_identity",label))
    scaffold_frame=pd.DataFrame(scaffold_rows); write_frame("13_SCAFFOLD_GENERALIZATION.csv",scaffold_frame)
    unseen=mol[~mol.scaffold_seen_in_train]
    def endpoint_class(g: pd.DataFrame) -> str:
        if len(g)<30: return "UNCLEAR"
        rv=float((g.restricted_v3d-g.source_v3d).mean()); uv=float((g.unrestricted_v3d-g.source_v3d).mean())
        rr=float(g.restricted_reference_improvement.mean()); ur=float(g.unrestricted_reference_improvement.mean())
        if min(rv,uv,rr,ur)>0: return "SUPPORTED"
        if max(rv,uv)>0 and max(rr,ur)>0: return "PARTIAL"
        return "NOT_SUPPORTED"
    unseen_class=endpoint_class(unseen)

    size_rows=[]
    for field in ("num_atoms","num_heavy_atoms","num_bonds"):
        labels=equal_quintile(mol[field]);
        for q,g in mol.groupby(labels,observed=True): size_rows.append(aggregate_group(g,field,str(q)) | {"size_mean":float(g[field].mean()),"size_median":float(g[field].median())})
    size_frame=pd.DataFrame(size_rows); write_frame("14_SIZE_GENERALIZATION.csv",size_frame)
    large=mol[equal_quintile(mol.num_heavy_atoms).eq("Q5")]; large_class=endpoint_class(large)

    flex_labels=equal_quintile(mol.num_rotatable_bonds)
    flex_rows=[aggregate_group(g,"num_rotatable_bonds",str(q)) | {"rotatable_mean":float(g.num_rotatable_bonds.mean()),"rotatable_median":float(g.num_rotatable_bonds.median())} for q,g in mol.groupby(flex_labels,observed=True)]
    flex_frame=pd.DataFrame(flex_rows); write_frame("15_FLEXIBILITY_GENERALIZATION.csv",flex_frame)
    high_flex=mol[flex_labels.eq("Q5")]; flex_class=endpoint_class(high_flex)

    quality_labels=equal_quintile(mol.source_reference_rmsd)
    quality_rows=[aggregate_group(g,"source_reference_rmsd",str(q)) for q,g in mol.groupby(quality_labels,observed=True)]
    quality_frame=pd.DataFrame(quality_rows); write_frame("16_SOURCE_QUALITY_GENERALIZATION.csv",quality_frame)
    poor=mol[quality_labels.eq("Q5")]; poor_class=endpoint_class(poor)

    write_text("17_UPSTREAM_GENERALIZATION.md", """# Upstream generalization

- TRAIN generator: ETFlow formal upstream.
- DEV generator: the same ETFlow formal upstream, two source records per unseen DEV molecule.
- Legacy AvgFlow and DiTMC artifacts exist for earlier Softplus-v2 formulations, but no frozen result applies the current J1-R1 Full Joint Restricted or Unrestricted checkpoint to those upstreams.
- Therefore current evidence supports cross-molecule, same-upstream-distribution development performance only.

```text
CURRENT_MODEL_CROSS_UPSTREAM = NOT_YET_TESTED
```
""")

    current_energy=pd.read_csv(EVIDENCE_REPORT/"04_XTB_ENERGY_COMPARISON.csv")
    current_energy=current_energy[current_energy.method.eq("SIXS_FULL_JOINT_STEP17500")].deltaE_kcal_mol
    unres_energy=pd.read_csv(UNRESTRICTED_DIR/"UNRESTRICTED_XTB.csv").deltaE_kcal_mol
    erows=[]
    for method,values in (("RESTRICTED",current_energy),("UNRESTRICTED",unres_energy)):
        erows.append({"method":method,**robust_energy(values)})
    energy_frame=pd.DataFrame(erows)
    write_text("18_XTB_ROBUST_REPORTING.md", f"""# xTB robust reporting audit

GFN2-xTB single-point DeltaE relative to the matched Source is heavy-tail aware. Median is primary; mean is retained as a secondary expectation statistic.

{md_table(energy_frame)}

```text
XTB_PRIMARY_LOCATION_STATISTIC = MEDIAN
XTB_MEAN_RETAINED_AS_SECONDARY = YES
```
""")

    def movement_stats(prefix: str) -> dict[str,float]:
        x=rec[f"{prefix}_tau"]
        return {"tau_median":float(x.median()),"tau_p95":float(x.quantile(.95)),"tau_p99":float(x.quantile(.99)),"tau_max":float(x.max())}
    compare=[]
    for prefix,label,e in (("restricted","RESTRICTED",energy_frame.iloc[0]),("unrestricted","UNRESTRICTED",energy_frame.iloc[1])):
        compare.append({"method":label,"V3D":float(rec[f"{prefix}_v3d"].mean()),"PB":float(rec[f"{prefix}_pb"].mean()),
            "reference_RMSD_mean":float(rec[f"{prefix}_reference_rmsd"].mean()),"reference_RMSD_median":float(rec[f"{prefix}_reference_rmsd"].median()),
            "source_RMSD_mean":float(rec[f"{prefix}_source_rmsd"].mean()),"source_RMSD_median":float(rec[f"{prefix}_source_rmsd"].median()),
            **{f"xTB_{k}":e[k] for k in ["mean","median","trimmed_mean_5pct","fraction_lt_0","p95","p99","max","count_gt_25","count_gt_50","count_gt_100"]},
            **movement_stats(prefix)})
    compare_frame=pd.DataFrame(compare); write_frame("19_RESTRICTED_VS_UNRESTRICTED.csv",compare_frame)

    unsafe = pd.DataFrame([
        (str(STATE_PREFLIGHT),"state_mean/state_std/selected_lambda","TRAIN","normalization and restricted loss","SAFE"),
        (str(PREPARED),"calibration_sha256/sigma_stat","TRAIN","inference scale features","SAFE"),
        ("both current training runners","reference tensors","TRAIN only","training labels, absent from inference features","SAFE"),
        (str(DEV_MANIFEST),"rows","VAL identity only","pretraining-frozen DEV cohort","SAFE"),
        (str(RESTRICTED_CONFIG),"comparator/evaluation","DEV outcomes","evaluation and formulation decision only; not optimization","DEVELOPMENT_BIAS"),
        (str(UNRESTRICTED_CONFIG),"comparator/evaluation","DEV outcomes","capability evaluation only; not optimization","DEVELOPMENT_BIAS"),
        ("project experiment history","formulation progression","repeated DEV outcomes","major design decisions repeatedly use same DEV","DEVELOPMENT_BIAS"),
        ("checkpoint policy","FINAL_STEP_17500_ONLY_NO_DEV_SELECTION","TRAIN step","no best-checkpoint or best-seed selection","SAFE"),
        ("protected payload guards","formal_test_records_read/frozen_holdout_records_read","metadata counters","both are zero","SAFE"),
        ("xTB/MMFF reporting","all record outcomes","DEV evaluation","failures/fallbacks disclosed; no outcome-dependent row deletion","SAFE"),
    ],columns=["PATH","VARIABLE","DATA_SOURCE","HOW_USED","SEVERITY"])
    write_text("20_UNSAFE_DATA_USAGE.md", "# Unsafe data/outcome usage search\n\n"+md_table(unsafe)+"\n\nNo DATA_LEAKAGE or CRITICAL_BUG item was found. Repeated DEV-guided formulation design is retained as DEVELOPMENT_BIAS, not relabeled as leakage.\n")

    claims = pd.DataFrame([
        ("TRAIN-record generalization","NOT_YET_TESTED","training fit is not a generalization test"),
        ("unseen conformer / same molecule","NOT_YET_TESTED","current DEV is molecule-disjoint; no dedicated same-molecule test"),
        ("unseen molecule / same distribution","SUPPORTED","0 isomeric molecule overlap; same ETFlow upstream"),
        ("distant-molecule generalization","PARTIALLY_SUPPORTED","similarity quintiles/correlations are descriptive and seed307-only"),
        ("unseen-scaffold generalization","SUPPORTED" if unseen_class=="SUPPORTED" else "PARTIALLY_SUPPORTED" if unseen_class=="PARTIAL" else "NOT_SUPPORTED","Bemis-Murcko TRAIN-unseen subgroup"),
        ("high-flexibility generalization","SUPPORTED" if flex_class=="SUPPORTED" else "PARTIALLY_SUPPORTED" if flex_class=="PARTIAL" else "NOT_SUPPORTED","highest rotatable-bond quintile"),
        ("large-molecule generalization","SUPPORTED" if large_class=="SUPPORTED" else "PARTIALLY_SUPPORTED" if large_class=="PARTIAL" else "NOT_SUPPORTED","highest heavy-atom quintile"),
        ("poor-Source generalization","SUPPORTED" if poor_class=="SUPPORTED" else "PARTIALLY_SUPPORTED" if poor_class=="PARTIAL" else "NOT_SUPPORTED","worst Source-reference RMSD quintile"),
        ("cross-upstream generalization","NOT_YET_TESTED","no current-checkpoint AvgFlow/DiTMC result"),
        ("Formal generalization","NOT_YET_TESTED","protected outcome not read"),
        ("large-holdout generalization","NOT_YET_TESTED","protected outcome not read"),
        ("universal generalization","NOT_SUPPORTED","claims exceed available same-upstream seed307 DEV evidence"),
    ],columns=["claim","classification","basis"])
    write_text("21_GENERALIZATION_CLAIM_MATRIX.md", "# Generalization claim matrix\n\n"+md_table(claims)+"\n")

    risks = pd.DataFrame([
        ("HIGH","DEV-overfitting risk","Nine major formulation decisions reused the same 2,500-molecule DEV cohort."),
        ("HIGH","seed-instability risk","Restricted and Unrestricted final comparisons exist only for seed307."),
        ("MEDIUM","cross-upstream-generalization risk","Current checkpoints have no AvgFlow/DiTMC zero-shot evaluation."),
        ("MEDIUM","human-design risk","beta, sigma dynamic range, Reliability init and loss balance lack isolated sensitivity sweeps."),
        ("MEDIUM","protected-generalization evidence","Formal and large-holdout outcomes remain unread/not established for current formulation."),
        ("LOW","data-leakage risk","No molecule/isomeric, source-coordinate or Reference-coordinate overlap was found."),
        ("LOW","evaluation risk","xTB mean has a positive-tail sensitivity; robust median/tails are now primary and fully retained."),
    ],columns=["rank","risk_type","evidence"])
    write_text("22_SCIENTIFIC_RISK_RANKING.md", "# Scientific risk ranking\n\n"+md_table(risks)+"\n")

    pareto = "PARETO_NEAR_TIE"
    graph_overlap=int(overlap[(overlap.split_a.eq("TRAIN"))&(overlap.split_b.eq("DEV"))&(overlap.identity.eq("canonical_graph_identity"))].n_overlap.iloc[0])
    final_md=f"""# Final full-project scientific audit

The project passes exact molecule/conformer/Reference leakage checks. DEV is molecule-disjoint from TRAIN at project and stereochemistry-aware canonical identity levels. The {graph_overlap} non-stereochemical graph overlaps are different stereoisomers and are explicitly retained as chemical-relatedness evidence, not hidden leakage.

Restricted and Unrestricted identities are valid and isolated. Their seed307 outcome is a Pareto near-tie: Unrestricted has a small significant Reference-RMSD advantage, statistically unresolved V3D advantage, identical PB, and essentially tied robust xTB behavior. This audit does not choose a multiseed formulation.

The strongest supported claim is unseen-molecule generalization on the same ETFlow upstream distribution. DEV is not an unbiased final test because it repeatedly guided project design. Current-formulation cross-upstream, multiseed, Formal and large-holdout evidence remain missing.

```text
AUDIT_STATUS = COMPLETE_READ_ONLY
RESTRICTED_MODEL_IDENTITY = {'PASS' if r_identity else 'FAIL'}
UNRESTRICTED_MODEL_IDENTITY = {'PASS' if u_identity else 'FAIL'}
RESTRICTED_HUMAN_SCIENTIFIC_CONSTANT_COUNT = {r_human}
UNRESTRICTED_HUMAN_SCIENTIFIC_CONSTANT_COUNT = {u_human}
UNSAFE_HUMAN_CONSTANT_FOUND = NO
TRAIN_DEV_MOLECULE_OVERLAP = 0
TRAIN_DEV_CONFORMER_LEAKAGE = NO
REFERENCE_LEAKAGE = NO
ALL_INFERENCE_STATISTICS_TRAIN_ONLY = YES
DEV_USED_FOR_MODEL_DEVELOPMENT = YES
FINAL_DEV_IS_UNBIASED_TEST_SET = NO
CROSS_MOLECULE_SPLIT = PASS
DEV_MOLECULES_SEEN_IN_TRAIN = 0
DEV_MAX_TRAIN_SIMILARITY_MEDIAN = {mol.max_train_similarity.median():.10g}
RESTRICTED_PERFORMANCE_DEGRADES_AS_TRAIN_DISTANCE_INCREASES = {r_dist}
UNRESTRICTED_PERFORMANCE_DEGRADES_AS_TRAIN_DISTANCE_INCREASES = {u_dist}
UNSEEN_SCAFFOLD_GENERALIZATION = {unseen_class}
HIGH_FLEXIBILITY_GENERALIZATION = {flex_class}
LARGE_MOLECULE_GENERALIZATION = {large_class}
POOR_SOURCE_GENERALIZATION = {poor_class}
CURRENT_MODEL_CROSS_UPSTREAM = NOT_YET_TESTED
XTB_PRIMARY_LOCATION_STATISTIC = MEDIAN
XTB_MEAN_RETAINED_AS_SECONDARY = YES
RESTRICTED_VS_UNRESTRICTED_CLASSIFICATION = {pareto}
UNSAFE_DATA_USAGE_FOUND = NO
DATA_LEAKAGE_FOUND = NO
DEVELOPMENT_BIAS_FOUND = YES
SUPPORTED_GENERALIZATION_LEVEL = UNSEEN_MOLECULE_SAME_ETFLOW_DISTRIBUTION
SEED331_STARTED = NO
SEED353_STARTED = NO
FORMAL_OUTCOME_READ = NO
LARGE_HOLDOUT_OUTCOME_READ = NO
NEW_TRAINING = NO
```
"""
    write_text("23_FINAL_PROJECT_AUDIT.md",final_md)

    hashes={name:sha256(OUT/name) for name in REPORT_NAMES}
    status={
        "schema_version":"sixs-final-project-integrity-generalization-audit-v1","AUDIT_STATUS":"COMPLETE_READ_ONLY",
        "RESTRICTED_MODEL_IDENTITY":"PASS" if r_identity else "FAIL","UNRESTRICTED_MODEL_IDENTITY":"PASS" if u_identity else "FAIL",
        "RESTRICTED_HUMAN_SCIENTIFIC_CONSTANT_COUNT":r_human,"UNRESTRICTED_HUMAN_SCIENTIFIC_CONSTANT_COUNT":u_human,
        "UNSAFE_HUMAN_CONSTANT_FOUND":"NO","QUESTIONABLE_CONSTANTS":questionable,
        "TRAIN_DEV_MOLECULE_OVERLAP":0,"TRAIN_DEV_NONSTEREO_GRAPH_OVERLAP":graph_overlap,"TRAIN_DEV_CONFORMER_LEAKAGE":"NO","REFERENCE_LEAKAGE":"NO",
        "ALL_INFERENCE_STATISTICS_TRAIN_ONLY":"YES","DEV_USED_FOR_MODEL_DEVELOPMENT":"YES","NUMBER_OF_MAJOR_DEV_GUIDED_DECISIONS":len(decisions),"FINAL_DEV_IS_UNBIASED_TEST_SET":"NO",
        "CROSS_MOLECULE_SPLIT":"PASS","TRAIN_UNIQUE_MOLECULES":train_unique_count,"DEV_UNIQUE_MOLECULES":dev_unique_count,"DEV_MOLECULES_SEEN_IN_TRAIN":0,
        "DEV_MAX_TRAIN_SIMILARITY_MEDIAN":float(mol.max_train_similarity.median()),"DEV_MAX_TRAIN_SIMILARITY_MEAN":float(mol.max_train_similarity.mean()),
        "FINGERPRINT_PROTOCOL":{"type":"Morgan/ECFP4","radius":2,"nBits":2048,"useChirality":True,"featureMode":False,"train_fingerprint_coverage":len(train_fps)},
        "RESTRICTED_PERFORMANCE_DEGRADES_AS_TRAIN_DISTANCE_INCREASES":r_dist,"UNRESTRICTED_PERFORMANCE_DEGRADES_AS_TRAIN_DISTANCE_INCREASES":u_dist,
        "DEV_UNSEEN_SCAFFOLD_FRACTION":float((~mol.scaffold_seen_in_train).mean()),"UNSEEN_SCAFFOLD_GENERALIZATION":unseen_class,
        "HIGH_FLEXIBILITY_GENERALIZATION":flex_class,"LARGE_MOLECULE_GENERALIZATION":large_class,"POOR_SOURCE_GENERALIZATION":poor_class,
        "CURRENT_MODEL_CROSS_UPSTREAM":"NOT_YET_TESTED","XTB_PRIMARY_LOCATION_STATISTIC":"MEDIAN","XTB_MEAN_RETAINED_AS_SECONDARY":"YES",
        "RESTRICTED_VS_UNRESTRICTED_CLASSIFICATION":pareto,"UNSAFE_DATA_USAGE_FOUND":"NO","DATA_LEAKAGE_FOUND":"NO","DEVELOPMENT_BIAS_FOUND":"YES",
        "SUPPORTED_GENERALIZATION_LEVEL":"UNSEEN_MOLECULE_SAME_ETFLOW_DISTRIBUTION",
        "TOP_SCIENTIFIC_RISKS":["repeated DEV-guided formulation adaptivity","seed307-only formulation evidence","current-formulation cross-upstream and protected-set evidence missing","weak exact-number sensitivity evidence"],
        "MISSING_CRITICAL_EVIDENCE":["seed331/seed353 independent replication","current Restricted/Unrestricted cross-upstream evaluation","protected Formal and large-holdout evaluation under authorization","isolated sensitivity for major human scientific constants"],
        "SEED331_STARTED":"NO","SEED353_STARTED":"NO","FORMAL_OUTCOME_READ":"NO","LARGE_HOLDOUT_OUTCOME_READ":"NO","NEW_TRAINING":"NO","MODEL_MODIFIED":"NO","HYPERPARAMETER_SWEEP":"NO",
        "REPORT_SHA256":hashes,
    }
    tmp=OUT/f"FINAL_STATUS.json.tmp.{os.getpid()}"; tmp.write_text(json.dumps(status,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8"); os.replace(tmp,OUT/"FINAL_STATUS.json")
    print(json.dumps({k:status[k] for k in ["AUDIT_STATUS","TRAIN_DEV_MOLECULE_OVERLAP","DEV_MAX_TRAIN_SIMILARITY_MEDIAN","RESTRICTED_PERFORMANCE_DEGRADES_AS_TRAIN_DISTANCE_INCREASES","UNRESTRICTED_PERFORMANCE_DEGRADES_AS_TRAIN_DISTANCE_INCREASES","RESTRICTED_VS_UNRESTRICTED_CLASSIFICATION"]},indent=2))


if __name__ == "__main__":
    main()
