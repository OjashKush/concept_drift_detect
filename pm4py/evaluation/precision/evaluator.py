from pm4py.evaluation.precision.variants import etconformance_token, align_etconformance
from pm4py.objects.conversion.log import converter as log_conversion
from pm4py.objects.petri.check_soundness import check_easy_soundness_net_in_fin_marking
from enum import Enum
from pm4py.util import exec_utils


class Variants(Enum):
    ETCONFORMANCE_TOKEN = etconformance_token
    ALIGN_ETCONFORMANCE = align_etconformance


ETCONFORMANCE_TOKEN = Variants.ETCONFORMANCE_TOKEN
ALIGN_ETCONFORMANCE = Variants.ALIGN_ETCONFORMANCE

VERSIONS = {ETCONFORMANCE_TOKEN, ALIGN_ETCONFORMANCE}


def apply(log, net, marking, final_marking, parameters=None, variant=None):
    """
    Method to apply ET Conformance

    Parameters
    -----------
    log
        Trace log
    net
        Petri net
    marking
        Initial marking
    final_marking
        Final marking
    parameters
        Parameters of the algorithm, including:
            pm4py.util.constants.PARAMETER_CONSTANT_ACTIVITY_KEY -> Activity key
    variant
        Variant of the algorithm that should be applied:
            - Variants.ETCONFORMANCE_TOKEN
            - Variants.ALIGN_ETCONFORMANCE
    """
    if parameters is None:
        parameters = {}

    log = log_conversion.apply(log, parameters, log_conversion.TO_EVENT_LOG)

    # execute the following part of code when the variant is not specified by the user
    if variant is None:
        if not (check_easy_soundness_net_in_fin_marking(
                net,
                marking,
                final_marking)):
            # in the case the net is not a easy sound workflow net, we must apply token-based replay
            variant = ETCONFORMANCE_TOKEN
        else:
            # otherwise, use the align-etconformance approach (safer, in the case the model contains duplicates)
            variant = ALIGN_ETCONFORMANCE

    return exec_utils.get_variant(variant).apply(log, net, marking,
                             final_marking, parameters=parameters)
