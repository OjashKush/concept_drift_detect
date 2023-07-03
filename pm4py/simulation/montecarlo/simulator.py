from pm4py.simulation.montecarlo.variants import petri_semaph_fifo
from pm4py.simulation.montecarlo.outputs import Outputs
from pm4py.util import exec_utils
from enum import Enum


class Variants(Enum):
    PETRI_SEMAPH_FIFO = petri_semaph_fifo


DEFAULT_VARIANT = Variants.PETRI_SEMAPH_FIFO

VERSIONS = {Variants.PETRI_SEMAPH_FIFO}


def apply(log, net, im, fm, variant=DEFAULT_VARIANT, parameters=None):
    """
    Performs a Monte Carlo simulation of an accepting Petri net without duplicate transitions and where the preset is always
    distinct from the postset

    Parameters
    -------------
    log
        Event log
    net
        Accepting Petri net without duplicate transitions and where the preset is always distinct from the postset
    im
        Initial marking
    fm
        Final marking
    variant
        Variant of the algorithm to use:
        - Variants.PETRI_SEMAPH_FIFO
    parameters
        Parameters of the algorithm:
            Parameters.PARAM_NUM_SIMULATIONS => (default: 100)
            Parameters.PARAM_FORCE_DISTRIBUTION => Force a particular stochastic distribution (e.g. normal) when the stochastic map
            is discovered from the log (default: None; no distribution is forced)
            Parameters.PARAM_ENABLE_DIAGNOSTICS => Enable the printing of diagnostics (default: True)
            Parameters.PARAM_DIAGN_INTERVAL => Interval of time in which diagnostics of the simulation are printed (default: 32)
            Parameters.PARAM_CASE_ARRIVAL_RATIO => Case arrival of new cases (default: None; inferred from the log)
            Parameters.PARAM_PROVIDED_SMAP => Stochastic map that is used in the simulation (default: None; inferred from the log)
            Parameters.PARAM_MAP_RESOURCES_PER_PLACE => Specification of the number of resources available per place
            (default: None; each place gets the default number of resources)
            Parameters.PARAM_DEFAULT_NUM_RESOURCES_PER_PLACE => Default number of resources per place when not specified
            (default: 1; each place gets 1 resource and has to wait for the resource to finish)
            Parameters.PARAM_SMALL_SCALE_FACTOR => Scale factor for the sleeping time of the actual simulation
            (default: 864000.0, 10gg)
            Parameters.PARAM_MAX_THREAD_EXECUTION_TIME => Maximum execution time per thread (default: 60.0, 1 minute)

    Returns
    ------------
    simulated_log
        Simulated event log
    simulation_result
        Result of the simulation:
            Outputs.OUTPUT_PLACES_INTERVAL_TREES => inteval trees that associate to each place the times in which it was occupied.
            Outputs.OUTPUT_TRANSITIONS_INTERVAL_TREES => interval trees that associate to each transition the intervals of time
            in which it could not fire because some token was in the output.
            Outputs.OUTPUT_CASES_EX_TIME => Throughput time of the cases included in the simulated log
            Outputs.OUTPUT_MEDIAN_CASES_EX_TIME => Median of the throughput times
            Outputs.OUTPUT_CASE_ARRIVAL_RATIO => Case arrival ratio that was specified in the simulation
            Outputs.OUTPUT_TOTAL_CASES_TIME => Total time occupied by cases of the simulated log
    """
    return exec_utils.get_variant(variant).apply(log, net, im, fm, parameters=parameters)
