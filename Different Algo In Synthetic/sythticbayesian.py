import numpy as np
import scipy.stats as stats
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from pm4py.statistics.time_series import construct
from pm4py.statistics.time_series import change_points
from pm4py.statistics.time_series import cause_effect

def bayesian_changepoint_detection(data):
    """
    This function implements Bayesian changepoint detection.

    Args:
        data: A list of data points.

    Returns:
        A list of changepoints.
    """

    # Initialize the model.
    model = np.ones(len(data))

    # Iterate over the data and find the most likely changepoints.
    changepoints = []
    for i in range(len(data)):
        if model[i] < 0.5:
            changepoints.append(i)
        model[i] *= (1 - model[i - 1])

    return changepoints

def main():
    logs = construct.subdivide_log(
        'pm4py/statistics/time_series/experiments/data/Synthetic_Insurance_Claim_large.xes',
        datetime(1970, 1, 15), datetime(1970, 10, 27), 1
    )

    pen_primary = 3
    pen_secondary = 1.5

    primary_names, primary_features = construct.apply_feature_extraction(
        logs, ["direct_follows_relations"]
    )

    # Apply Bayesian changepoint detection to find changepoints in the primary features
    cp_1 = bayesian_changepoint_detection(primary_features)

    secondary_names, secondary_features = construct.apply_feature_extraction(
        logs, ["all_numeric_data"]
    )

    # Apply Bayesian changepoint detection to find changepoints in the secondary features
    cp_2 = bayesian_changepoint_detection(secondary_features)

    res = cause_effect.granger_causality(
        primary_features, secondary_features,
        secondary_names, cp_1, cp_2,
        p_value=0.0000000000001
    )

    cause_effect.draw_ca(
        res, primary_features, primary_names,
        secondary_features, secondary_names,
        store_path="synthetic.pdf"
    )

if __name__ == "__main__":
    # execute only if run as a script
    main()
