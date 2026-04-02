"""
Contraction helpers.

For your current MVP (DMRG/MPS), you don't need these.
They become useful when you add:
- full circuit tensor network contraction (cotengra)
- custom contraction paths
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ContractionReport:
    optimize: str
    backend: Optional[str] = None
    note: str = ""


def contract_tn(tn: Any, optimize: str = "auto-hq", backend: str | None = None):
    """
    Contract a quimb tensor network with an explicit optimizer.

    optimize examples:
      - "auto"
      - "auto-hq"
      - "greedy"
      - a cotengra optimizer object (advanced)
    """
    kwargs = {"optimize": optimize}
    if backend is not None:
        kwargs["backend"] = backend
    return tn.contract(all, **kwargs)


def describe_optimizer(optimize: str, backend: str | None = None) -> ContractionReport:
    note = ""
    if optimize in ("auto", "auto-hq"):
        note = "quimb chooses an optimizer automatically (auto-hq tends to spend more effort optimizing)."
    elif optimize == "greedy":
        note = "greedy contraction ordering (fast to choose, not always minimal cost)."
    else:
        note = "custom/advanced optimizer."
    return ContractionReport(optimize=optimize, backend=backend, note=note)