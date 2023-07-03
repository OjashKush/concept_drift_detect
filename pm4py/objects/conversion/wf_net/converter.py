from pm4py.objects.conversion.wf_net.variants import to_process_tree
from pm4py.util import exec_utils
from enum import Enum


class Variants(Enum):
    TO_PROCESS_TREE = to_process_tree


DEFAULT_VARIANT = Variants.TO_PROCESS_TREE


def apply(net, im, fm, variant=DEFAULT_VARIANT, parameters=None):
    return exec_utils.get_variant(variant).apply(net, im, fm, parameters=parameters)
