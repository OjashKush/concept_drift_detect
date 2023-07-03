from datetime import datetime
import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore", category=FutureWarning)
from pm4py.statistics.time_series import construct
from pm4py.statistics.time_series import change_points

def cusum(data, threshold):
    n = len(data)
    mean = np.mean(data, axis=0)
    cumulative_sum = np.zeros_like(data)
    drift_indices = []

    for i in range(1, n):
        cumulative_sum[i] = np.maximum(0, cumulative_sum[i-1] + data[i] - mean - threshold)
        if np.any(cumulative_sum[i-1] > 0) and np.any(cumulative_sum[i] > 0):
            drift_indices.append(i)

    return drift_indices




def determine_change_points_primary(cp_1):
    # Define and implement the logic to determine the change points for the primary perspective
    # Replace this with your own implementation or use appropriate methods from the pm4py library
    change_points_primary = cp_1  # Modify this to contain the actual change points
    return change_points_primary

def detect_drift_type(cp_1, cp_2):
    """
    Detects the type of drift in the given time series.

    Args:
        cp_1 (list): The change points for the primary time series.
        cp_2 (list): The change points for the secondary time series.

    Returns:
        str: The type of drift.
    """

    drift_type = None

    print("Change points for primary perspective:", cp_1)
    print("Change points for secondary perspective:", cp_2)

    if len(cp_1) == 1 and len(cp_2) == 1:
        if cp_1[0] == cp_2[0]:
            drift_type = "Sudden"
        else:
            drift_type = "Gradual"
    elif len(cp_1) > 1 and len(cp_2) == 1:
        drift_type = "Recurring"
    elif len(cp_1) == 1 and len(cp_2) > 1:
        drift_type = "Incremental"
    elif len(cp_1) > 1 and len(cp_2) > 1:
        drift_type = "Gradual"
    else:
        drift_type = "No drift"

    return drift_type

def main():
    logs = construct.subdivide_log(
        'pm4py/statistics/time_series/experiments/data/Synthetic_Insurance_Claim_large.xes',
        datetime(1970, 1, 15), datetime(1970, 10, 27), 1)

    pen_primary = 3
    pen_secondary = 1.5

    primary_names, primary_features = construct.apply_feature_extraction(
        logs, ["direct_follows_relations"])
    # dimensionality reduction with automated choice of dimensionality
    reduced_primary = construct.pca_reduction(primary_features, 'mle',
                                              normalize=True,
                                              normalize_function="max")
    cp_1 = change_points.rpt_pelt(reduced_primary, pen=pen_primary)
    cp_1 = cp_1[:len(reduced_primary)]
    print(cp_1)

    secondary_names, secondary_features = construct.apply_feature_extraction(logs,
                                                                             ["all_numeric_data"])
    # dimensionality reduction with automated choice of dimensionality
    reduced_secondary = construct.pca_reduction(secondary_features,
                                                'mle',
                                                normalize=True,
                                                normalize_function="max")
    cp_2 = change_points.rpt_pelt(reduced_secondary, pen=pen_secondary)
    cp_2 = cp_2[:len(reduced_secondary)]
    print(cp_2)

    # Apply CUSUM algorithm for drift detection
    threshold = 0.9  # Adjust the threshold value as needed
    drift_indices_cusum = cusum(reduced_primary, threshold)
    print("Drift indices detected by CUSUM algorithm:", drift_indices_cusum)

    # Detect the type of drift
    change_points_primary = determine_change_points_primary(cp_1)  # Determine change points for the primary perspective
    drift_type = detect_drift_type(change_points_primary, cp_2)
    print("Drift type:", drift_type)


if __name__ == "__main__":
    # execute only if run as a script
    main()
