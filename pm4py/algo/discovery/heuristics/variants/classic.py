import pkgutil
from copy import deepcopy

from pm4py.algo.discovery.dfg import algorithm as dfg_alg
from pm4py.algo.discovery.heuristics.parameters import Parameters
from pm4py.objects.conversion.heuristics_net import converter as hn_conv_alg
from pm4py.objects.heuristics_net import defaults
from pm4py.objects.heuristics_net.net import HeuristicsNet
from pm4py.statistics.attributes.log import get as log_attributes
from pm4py.statistics.end_activities.log import get as log_ea_filter
from pm4py.statistics.start_activities.log import get as log_sa_filter
from pm4py.util import constants
from pm4py.util import exec_utils
from pm4py.util import xes_constants as xes


def apply(log, parameters=None):
    """
    Discovers a Petri net using Heuristics Miner

    Parameters
    ------------
    log
        Event log
    parameters
        Possible parameters of the algorithm,
        including:
            - Parameters.ACTIVITY_KEY
            - Parameters.TIMESTAMP_KEY
            - Parameters.CASE_ID_KEY
            - Parameters.DEPENDENCY_THRESH
            - Parameters.AND_MEASURE_THRESH
            - Parameters.MIN_ACT_COUNT
            - Parameters.MIN_DFG_OCCURRENCES
            - Parameters.DFG_PRE_CLEANING_NOISE_THRESH
            - Parameters.LOOP_LENGTH_TWO_THRESH

    Returns
    ------------
    net
        Petri net
    im
        Initial marking
    fm
        Final marking
    """
    if parameters is None:
        parameters = {}

    heu_net = apply_heu(log, parameters=parameters)
    net, im, fm = hn_conv_alg.apply(heu_net, parameters=parameters)

    return net, im, fm


def apply_pandas(df, parameters=None):
    """
    Discovers a Petri net using Heuristics Miner

    Parameters
    ------------
    df
        Pandas dataframe
    parameters
        Possible parameters of the algorithm,
        including: activity_key, case_id_glue, timestamp_key,
        dependency_thresh, and_measure_thresh, min_act_count, min_dfg_occurrences, dfg_pre_cleaning_noise_thresh,
        loops_length_two_thresh

    Returns
    ------------
    net
        Petri net
    im
        Initial marking
    fm
        Final marking
    """
    if parameters is None:
        parameters = {}

    if pkgutil.find_loader("pandas"):
        activity_key = exec_utils.get_param_value(Parameters.ACTIVITY_KEY, parameters, xes.DEFAULT_NAME_KEY)
        case_id_glue = exec_utils.get_param_value(Parameters.CASE_ID_KEY, parameters, constants.CASE_CONCEPT_NAME)
        start_timestamp_key = exec_utils.get_param_value(Parameters.START_TIMESTAMP_KEY, parameters,
                                                         None)
        timestamp_key = exec_utils.get_param_value(Parameters.TIMESTAMP_KEY, parameters, xes.DEFAULT_TIMESTAMP_KEY)

        from pm4py.algo.discovery.dfg.adapters.pandas import df_statistics, freq_triples as get_freq_triples
        from pm4py.statistics.attributes.pandas import get as pd_attributes
        from pm4py.statistics.start_activities.pandas import get as pd_sa_filter
        from pm4py.statistics.end_activities.pandas import get as pd_ea_filter

        start_activities = pd_sa_filter.get_start_activities(df, parameters=parameters)
        end_activities = pd_ea_filter.get_end_activities(df, parameters=parameters)
        activities_occurrences = pd_attributes.get_attribute_values(df, activity_key, parameters=parameters)
        activities = list(activities_occurrences.keys())
        if timestamp_key in df:
            dfg = df_statistics.get_dfg_graph(df, case_id_glue=case_id_glue,
                                              activity_key=activity_key, timestamp_key=timestamp_key,
                                              start_timestamp_key=start_timestamp_key)
            dfg_window_2 = df_statistics.get_dfg_graph(df, case_id_glue=case_id_glue,
                                                       activity_key=activity_key, timestamp_key=timestamp_key, window=2,
                                                       start_timestamp_key=start_timestamp_key)
            frequency_triples = get_freq_triples.get_freq_triples(df, case_id_glue=case_id_glue,
                                                                  activity_key=activity_key,
                                                                  timestamp_key=timestamp_key)

        else:
            dfg = df_statistics.get_dfg_graph(df, case_id_glue=case_id_glue,
                                              activity_key=activity_key, sort_timestamp_along_case_id=False)
            dfg_window_2 = df_statistics.get_dfg_graph(df, case_id_glue=case_id_glue,
                                                       activity_key=activity_key, sort_timestamp_along_case_id=False,
                                                       window=2)
            frequency_triples = get_freq_triples.get_freq_triples(df, case_id_glue=case_id_glue,
                                                                  activity_key=activity_key,
                                                                  timestamp_key=timestamp_key,
                                                                  sort_timestamp_along_case_id=False)

        heu_net = apply_heu_dfg(dfg, activities=activities, activities_occurrences=activities_occurrences,
                                start_activities=start_activities, end_activities=end_activities,
                                dfg_window_2=dfg_window_2,
                                freq_triples=frequency_triples, parameters=parameters)
        net, im, fm = hn_conv_alg.apply(heu_net, parameters=parameters)

        return net, im, fm


def apply_dfg(dfg, activities=None, activities_occurrences=None, start_activities=None, end_activities=None,
              parameters=None):
    """
    Discovers a Petri net using Heuristics Miner

    Parameters
    ------------
    dfg
        Directly-Follows Graph
    activities
        (If provided) list of activities of the log
    activities_occurrences
        (If provided) dictionary of activities occurrences
    start_activities
        (If provided) dictionary of start activities occurrences
    end_activities
        (If provided) dictionary of end activities occurrences
    parameters
        Possible parameters of the algorithm,
        including:
            - Parameters.ACTIVITY_KEY
            - Parameters.TIMESTAMP_KEY
            - Parameters.CASE_ID_KEY
            - Parameters.DEPENDENCY_THRESH
            - Parameters.AND_MEASURE_THRESH
            - Parameters.MIN_ACT_COUNT
            - Parameters.MIN_DFG_OCCURRENCES
            - Parameters.DFG_PRE_CLEANING_NOISE_THRESH
            - Parameters.LOOP_LENGTH_TWO_THRESH

    Returns
    ------------
    net
        Petri net
    im
        Initial marking
    fm
        Final marking
    """
    if parameters is None:
        parameters = {}

    heu_net = apply_heu_dfg(dfg, activities=activities, activities_occurrences=activities_occurrences,
                            start_activities=start_activities, end_activities=end_activities, parameters=parameters)
    net, im, fm = hn_conv_alg.apply(heu_net, parameters=parameters)

    return net, im, fm


def apply_heu(log, parameters=None):
    """
    Discovers an Heuristics Net using Heuristics Miner

    Parameters
    ------------
    log
        Event log
    parameters
        Possible parameters of the algorithm,
        including:
            - Parameters.ACTIVITY_KEY
            - Parameters.TIMESTAMP_KEY
            - Parameters.CASE_ID_KEY
            - Parameters.DEPENDENCY_THRESH
            - Parameters.AND_MEASURE_THRESH
            - Parameters.MIN_ACT_COUNT
            - Parameters.MIN_DFG_OCCURRENCES
            - Parameters.DFG_PRE_CLEANING_NOISE_THRESH
            - Parameters.LOOP_LENGTH_TWO_THRESH

    Returns
    ------------
    heu
        Heuristics Net
    """
    if parameters is None:
        parameters = {}

    activity_key = exec_utils.get_param_value(Parameters.ACTIVITY_KEY, parameters, xes.DEFAULT_NAME_KEY)

    start_activities = log_sa_filter.get_start_activities(log, parameters=parameters)
    end_activities = log_ea_filter.get_end_activities(log, parameters=parameters)
    activities_occurrences = log_attributes.get_attribute_values(log, activity_key, parameters=parameters)
    activities = list(activities_occurrences.keys())
    dfg = dfg_alg.apply(log, parameters=parameters)
    parameters_w2 = deepcopy(parameters)
    parameters_w2["window"] = 2
    dfg_window_2 = dfg_alg.apply(log, parameters=parameters_w2)
    freq_triples = dfg_alg.apply(log, parameters=parameters, variant=dfg_alg.Variants.FREQ_TRIPLES)

    return apply_heu_dfg(dfg, activities=activities, activities_occurrences=activities_occurrences,
                         start_activities=start_activities,
                         end_activities=end_activities, dfg_window_2=dfg_window_2, freq_triples=freq_triples,
                         parameters=parameters)


def apply_heu_dfg(dfg, activities=None, activities_occurrences=None, start_activities=None, end_activities=None,
                  dfg_window_2=None, freq_triples=None, parameters=None):
    """
    Discovers an Heuristics Net using Heuristics Miner

    Parameters
    ------------
    dfg
        Directly-Follows Graph
    activities
        (If provided) list of activities of the log
    activities_occurrences
        (If provided) dictionary of activities occurrences
    start_activities
        (If provided) dictionary of start activities occurrences
    end_activities
        (If provided) dictionary of end activities occurrences
    dfg_window_2
        (If provided) DFG of window 2
    freq_triples
        (If provided) Frequency triples
    parameters
        Possible parameters of the algorithm,
        including:
            - Parameters.ACTIVITY_KEY
            - Parameters.TIMESTAMP_KEY
            - Parameters.CASE_ID_KEY
            - Parameters.DEPENDENCY_THRESH
            - Parameters.AND_MEASURE_THRESH
            - Parameters.MIN_ACT_COUNT
            - Parameters.MIN_DFG_OCCURRENCES
            - Parameters.DFG_PRE_CLEANING_NOISE_THRESH
            - Parameters.LOOP_LENGTH_TWO_THRESH

    Returns
    ------------
    heu
        Heuristics Net
    """
    if parameters is None:
        parameters = {}

    dependency_thresh = exec_utils.get_param_value(Parameters.DEPENDENCY_THRESH, parameters,
                                                   defaults.DEFAULT_DEPENDENCY_THRESH)
    and_measure_thresh = exec_utils.get_param_value(Parameters.AND_MEASURE_THRESH, parameters,
                                                    defaults.DEFAULT_AND_MEASURE_THRESH)
    min_act_count = exec_utils.get_param_value(Parameters.MIN_ACT_COUNT, parameters, defaults.DEFAULT_MIN_ACT_COUNT)
    min_dfg_occurrences = exec_utils.get_param_value(Parameters.MIN_DFG_OCCURRENCES, parameters,
                                                     defaults.DEFAULT_MIN_DFG_OCCURRENCES)
    dfg_pre_cleaning_noise_thresh = exec_utils.get_param_value(Parameters.DFG_PRE_CLEANING_NOISE_THRESH, parameters,
                                                               defaults.DEFAULT_DFG_PRE_CLEANING_NOISE_THRESH)
    loops_length_two_thresh = exec_utils.get_param_value(Parameters.LOOP_LENGTH_TWO_THRESH, parameters,
                                                         defaults.DEFAULT_LOOP_LENGTH_TWO_THRESH)
    heu_net = HeuristicsNet(dfg, activities=activities, activities_occurrences=activities_occurrences,
                            start_activities=start_activities, end_activities=end_activities,
                            dfg_window_2=dfg_window_2,
                            freq_triples=freq_triples)
    heu_net.calculate(dependency_thresh=dependency_thresh, and_measure_thresh=and_measure_thresh,
                      min_act_count=min_act_count, min_dfg_occurrences=min_dfg_occurrences,
                      dfg_pre_cleaning_noise_thresh=dfg_pre_cleaning_noise_thresh,
                      loops_length_two_thresh=loops_length_two_thresh)

    return heu_net
