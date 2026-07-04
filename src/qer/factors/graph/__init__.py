"""Graph factors (feature classes A/B/C) built through the Subphase 3.2 engine.

Importing this subpackage registers the available graph factors into
``GRAPH_FACTORS`` -- a registry kept SEPARATE from the classical ``FACTORS`` so graph
features do not join the classical zoo or its multiple-testing denominator. The
harness (``compute_factor_ic``, ``factor_long_short``) evaluates a graph factor by
object, exactly as for a classical one.

Subphase 3.1 shipped :class:`PanelFactor`; Subphase 3.3 adds :class:`GraphFactor`,
:class:`DeltaGraphFactor`, and the correlation-network factors (feature class A).
"""

from qer.factors.graph.base import (  # noqa: F401
    DeltaGraphFactor,
    GraphFactor,
    PanelFactor,
    all_graph_factors,
    get_graph_factor,
    register_graph_factor,
)

# Import for the registration side-effect (populates GRAPH_FACTORS).
from qer.factors.graph import correlation_factors  # noqa: F401,E402

__all__ = [
    "PanelFactor",
    "GraphFactor",
    "DeltaGraphFactor",
    "register_graph_factor",
    "get_graph_factor",
    "all_graph_factors",
]
