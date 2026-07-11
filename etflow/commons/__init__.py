"""Public commons API with dependency-isolated lazy imports.

Historically this package imported chemistry, sampling, XTB, and CUDA extension
modules eagerly.  Lazy resolution preserves every public name while allowing
standalone geometry modules to run without unrelated optional dependencies.
"""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "build_conformer": ("covmat", "build_conformer"),
    "MoleculeFeaturizer": ("featurization", "MoleculeFeaturizer"),
    "apply_bond_jacobian": ("flexbond_jacobian", "apply_bond_jacobian"),
    "build_bond_local_frame": ("flexbond_jacobian", "build_bond_local_frame"),
    "identify_target_bonds": ("flexbond_jacobian", "identify_target_bonds"),
    "solve_q_star_least_squares": ("flexbond_jacobian", "solve_q_star_least_squares"),
    "get_base_data_dir": ("io", "get_base_data_dir"),
    "get_local_cache": ("io", "get_local_cache"),
    "load_json": ("io", "load_json"),
    "load_memmap": ("io", "load_memmap"),
    "load_npz": ("io", "load_npz"),
    "load_pkl": ("io", "load_pkl"),
    "save_memmap": ("io", "save_memmap"),
    "save_pkl": ("io", "save_pkl"),
    "batched_sampling": ("sample", "batched_sampling"),
    "kabsch_align": ("kabsch_utils", "kabsch_align"),
    "kabsch_rmsd": ("kabsch_utils", "kabsch_rmsd"),
    "select_best_reference_conformer": ("kabsch_utils", "select_best_reference_conformer"),
    "Queue": ("utils", "Queue"),
    "extend_graph_order_radius": ("utils", "extend_graph_order_radius"),
    "get_atomic_number_and_charge": ("utils", "get_atomic_number_and_charge"),
    "signed_volume": ("utils", "signed_volume"),
    "xtb_energy": ("xtb", "xtb_energy"),
    "xtb_optimize": ("xtb", "xtb_optimize"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()).union(__all__))
