import warnings
import re
import pandas as pd
from pm4py.statistics.time_series import construct
from pm4py.statistics.time_series import change_points
from datetime import datetime
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

def cusum(data, threshold):
    """
    Applies the CUSUM algorithm for drift detection.

    Args:
        data (numpy.ndarray): Time series data.
        threshold (float): Threshold value for drift detection.

    Returns:
        list: Indices of detected drift points.
    """
    n = len(data)
    mean = np.mean(data, axis=0)
    cumulative_sum = np.zeros_like(data)
    drift_indices = []

    for i in range(1, n):
        cumulative_sum[i] = np.maximum(0, cumulative_sum[i-1] + data[i] - mean - threshold)
        if np.any(cumulative_sum[i-1] > 0) and np.any(cumulative_sum[i] > 0):
            drift_indices.append(i)

    return drift_indices

def determine_change_points_primary():
    """
    Determines the change points for the primary perspective.

    Returns:
        list: Change points for the primary perspective.
    """
    # Define and implement the logic to determine the change points for the primary perspective
    # Replace this with your own implementation or use appropriate methods from the pm4py library
    cp_1 = [133]  # Modify this to contain the actual change points
    return cp_1

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

    print("Length of cp_1:", len(cp_1))
    print("Length of cp_2:", len(cp_2))

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

def detect_drift():
    # Load logs and subdivide them based on time intervals
    logs = construct.subdivide_log(
        'pm4py/statistics/time_series/experiments/data/BPI2019.xes',
        datetime(2017, 1, 1), datetime(2017, 12, 24), 7
    )

    # Preprocess logs to update lifecycle:transition attribute
    for log in logs:
        for trace in log:
            for event in trace:
                match = re.match(r"^.+\..+$", event["concept:name"])
                if match:
                    event["lifecycle:transition"] = match.group(0)
                else:
                    event["lifecycle:transition"] = event["concept:name"]

    pen_primary = 10
    pen_secondary = 5

    # Extract features and perform dimensionality reduction for the primary time series
    primary_names, primary_features = construct.apply_feature_extraction(
        logs, ["direct_follows_relations"]
    )

    # Convert primary_features to DataFrame
    primary_features = pd.DataFrame(primary_features)

    if primary_features.empty or pd.DataFrame(primary_names).empty:
        print("No primary features found. Check your data.")
        return

    if not any(primary_features.values.flatten()) or not any(pd.DataFrame(primary_names).values.flatten()):
        print("No primary features found. Check your data.")
        return

    if primary_features.ndim != 2:
        print("Invalid primary features data. Expected a 2D array.")
        return

    # Dimensionality reduction with automated choice of dimensionality using PCA
    reduced_primary = construct.pca_reduction(primary_features.to_numpy(), 'mle',
                                              normalize=True,
                                              normalize_function="max")

    # Flatten the reduced_primary array
    reduced_primary = np.array([item for sublist in reduced_primary for item in sublist])

    # Apply RPT-PELT algorithm for change point detection on the primary time series
    cp_1 = change_points.rpt_pelt(reduced_primary, pen=pen_primary)
    print(cp_1)

    # Extract features and perform dimensionality reduction for the secondary time series
    secondary_names, secondary_features = construct.apply_feature_extraction(
        logs, ["direct_follows_relations"]
    )

    # Dimensionality reduction with automated choice of dimensionality using PCA
    reduced_secondary = construct.pca_reduction(secondary_features,
                                                'mle',
                                                normalize=True,
                                                normalize_function="max")

    # Flatten the secondary features list
    secondary_features_flat = np.array([item for sublist in reduced_secondary for item in sublist])

    # Apply RPT-PELT algorithm for change point detection on the secondary time series
    cp_2 = change_points.rpt_pelt(secondary_features_flat, pen=pen_secondary)
    print(cp_2)

    # Apply CUSUM algorithm for drift detection on the primary time series
    threshold = 0.6  # Adjust the threshold value as needed
    drift_indices_cusum = cusum(reduced_primary, threshold)
    print("Drift indices detected by CUSUM algorithm:", drift_indices_cusum)

    # Determine change points for the primary perspective
    change_points_primary = determine_change_points_primary()

    # Detect the type of drift
    drift_type = detect_drift_type(change_points_primary, cp_2)
    print("Drift Type:", drift_type)


if __name__ == "__main__":
    detect_drift()
