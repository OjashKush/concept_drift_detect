from pm4py.objects.dfg.retrieval import log as log_calc


def apply(log, parameters=None):
    """
    Counts the number of directly follows occurrences, i.e. of the form <...a,b...>, in an event log.

    Parameters
    ----------
    log
        Trace log
    parameters
        Possible parameters passed to the algorithms:
            Parameters.ACTIVITY_KEY -> Attribute to use as activity

    Returns
    -------
    dfg
        DFG graph
    """
    return log_calc.native(log, parameters=parameters)
