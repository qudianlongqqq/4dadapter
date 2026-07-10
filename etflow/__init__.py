"""ETFlow package with lazy top-level imports for lightweight utilities."""

__all__ = ["BaseFlow"]


def __getattr__(name):
    if name == "BaseFlow":
        from etflow.models.model import BaseFlow

        return BaseFlow
    raise AttributeError(name)
