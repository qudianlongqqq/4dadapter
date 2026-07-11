"""Public model API with lazy imports for independent model namespaces."""

from importlib import import_module


_EXPORTS = {
    "BaseFlow": ("model", "BaseFlow"),
    "CartesianOptimizer": ("flexbond_optimizer", "CartesianOptimizer"),
    "FlexBond4DHybridOptimizer": ("flexbond_optimizer", "FlexBond4DHybridOptimizer"),
    "FlexBond4DOnlyOptimizer": ("flexbond_optimizer", "FlexBond4DOnlyOptimizer"),
    "FlexBondOptimizerLightningModule": ("flexbond_optimizer", "FlexBondOptimizerLightningModule"),
    "GatedKinematicFlowLightningModule": ("gated_kinematic_flow", "GatedKinematicFlowLightningModule"),
    "GlobalCoupled4DFlowLightningModule": ("global_coupled_4d_flow", "GlobalCoupled4DFlowLightningModule"),
    "build_motion_model": ("motion_factory", "build_motion_model"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()).union(__all__))
