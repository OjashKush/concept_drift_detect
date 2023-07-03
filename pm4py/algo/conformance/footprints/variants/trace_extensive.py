from pm4py.algo.discovery.footprints.outputs import Outputs
from enum import Enum
from pm4py.util import exec_utils, xes_constants, constants


class Parameters(Enum):
    CASE_ID_KEY = constants.PARAMETER_CONSTANT_CASEID_KEY
    ENABLE_ACT_ALWAYS_EXECUTED = "enable_act_always_executed"


class ConfOutputs(Enum):
    FOOTPRINTS = "footprints"
    START_ACTIVITIES = "start_activities"
    END_ACTIVITIES = "end_activities"
    ACTIVITIES_ALWAYS_HAPPENING = "activities_always_happening"
    MIN_LENGTH_FIT = "min_length_fit"
    IS_FOOTPRINTS_FIT = "is_footprints_fit"


def apply(log_footprints, model_footprints, parameters=None):
    """
    Apply footprints conformance between a log footprints object
    and a model footprints object

    Parameters
    -----------------
    log_footprints
        Footprints of the log (trace-by-trace)
    model_footprints
        Footprints of the model
    parameters
        Parameters of the algorithm

    Returns
    ------------------
    violations
        List containing, for each trace, a dictionary containing the violations
    """
    if parameters is None:
        parameters = {}

    if not type(log_footprints) is list:
        raise Exception(
            "it is possible to apply this variant only on trace-by-trace footprints, not overall log footprints!")

    enable_act_always_executed = exec_utils.get_param_value(Parameters.ENABLE_ACT_ALWAYS_EXECUTED, parameters, True)
    model_configurations = model_footprints[Outputs.SEQUENCE.value].union(model_footprints[Outputs.PARALLEL.value])

    ret = []
    for tr in log_footprints:
        trace_configurations = tr[Outputs.SEQUENCE.value].union(tr[Outputs.PARALLEL.value])
        trace_violations = {}
        trace_violations[ConfOutputs.FOOTPRINTS.value] = set(
            x for x in trace_configurations if x not in model_configurations)
        trace_violations[ConfOutputs.START_ACTIVITIES.value] = set(x for x in tr[Outputs.START_ACTIVITIES.value] if
                                                                   x not in model_footprints[
                                                                       Outputs.START_ACTIVITIES.value]) if Outputs.START_ACTIVITIES.value in model_footprints else set()
        trace_violations[ConfOutputs.END_ACTIVITIES.value] = set(
            x for x in tr[Outputs.END_ACTIVITIES.value] if x not in model_footprints[
                Outputs.END_ACTIVITIES.value]) if Outputs.END_ACTIVITIES.value in model_footprints else set()
        trace_violations[ConfOutputs.ACTIVITIES_ALWAYS_HAPPENING.value] = set(
            x for x in model_footprints[Outputs.ACTIVITIES_ALWAYS_HAPPENING.value] if x not in tr[
                Outputs.ACTIVITIES.value]) if Outputs.ACTIVITIES_ALWAYS_HAPPENING.value in model_footprints and enable_act_always_executed else set()
        trace_violations[ConfOutputs.MIN_LENGTH_FIT.value] = tr[Outputs.MIN_TRACE_LENGTH.value] >= model_footprints[
            Outputs.MIN_TRACE_LENGTH.value] if Outputs.MIN_TRACE_LENGTH.value in tr and Outputs.MIN_TRACE_LENGTH.value in model_footprints else True
        trace_violations[ConfOutputs.IS_FOOTPRINTS_FIT.value] = len(
            trace_violations[ConfOutputs.FOOTPRINTS.value]) == 0 and len(
            trace_violations[ConfOutputs.START_ACTIVITIES.value]) == 0 and len(
            trace_violations[ConfOutputs.END_ACTIVITIES.value]) == 0 and len(
            trace_violations[ConfOutputs.ACTIVITIES_ALWAYS_HAPPENING.value]) == 0 and trace_violations[
                                                                    ConfOutputs.MIN_LENGTH_FIT.value]

        ret.append(trace_violations)

    return ret


def get_diagnostics_dataframe(log, conf_result, parameters=None):
    """
    Gets the diagnostics dataframe from the log
    and the results of footprints conformance checking
    (trace-by-trace)

    Parameters
    --------------
    log
        Event log
    conf_result
        Conformance checking results (trace-by-trace)

    Returns
    --------------
    diagn_dataframe
        Diagnostics dataframe
    """
    if parameters is None:
        parameters = {}

    case_id_key = exec_utils.get_param_value(Parameters.CASE_ID_KEY, parameters, xes_constants.DEFAULT_TRACEID_KEY)

    import pandas as pd

    diagn_stream = []

    for index in range(len(log)):
        case_id = log[index].attributes[case_id_key]
        is_fit = conf_result[index][ConfOutputs.IS_FOOTPRINTS_FIT.value]
        footprints_violations = len(conf_result[index][ConfOutputs.FOOTPRINTS.value])
        start_activities_violations = len(conf_result[index][ConfOutputs.START_ACTIVITIES.value])
        end_activities_violations = len(conf_result[index][ConfOutputs.END_ACTIVITIES.value])
        act_always_happening_violations = len(conf_result[index][ConfOutputs.ACTIVITIES_ALWAYS_HAPPENING.value])
        min_length_fit = conf_result[index][ConfOutputs.MIN_LENGTH_FIT.value]

        diagn_stream.append({"case_id": case_id, "is_fit": is_fit, "footprints_violations": footprints_violations, "start_activities_violations": start_activities_violations,
                             "end_activities_violations": end_activities_violations, "act_always_happening_violations": act_always_happening_violations, "min_length_fit": min_length_fit})

    return pd.DataFrame(diagn_stream)
