from pm4py.simulation.tree_playout.variants import basic_playout, extensive, topbottom
from enum import Enum
from pm4py.util import exec_utils


class Variants(Enum):
    BASIC_PLAYOUT = basic_playout
    EXTENSIVE = extensive
    TOPBOTTOM = topbottom


DEFAULT_VARIANT = Variants.TOPBOTTOM


def apply(tree, variant=DEFAULT_VARIANT, parameters=None):
    """
    Performs a playout of a process tree

    Parameters
    ---------------
    tree
        Process tree
    variant
        Variant of the algorithm:
        - Variants.BASIC_PLAYOUT: basic playout
        - Variants.EXTENSIVE: extensive playout (all the possible traces)
    parameters
        Parameters of the algorithm
    """
    if parameters is None:
        parameters = {}

    return exec_utils.get_variant(variant).apply(tree, parameters=parameters)
