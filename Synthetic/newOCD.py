import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from pm4py.statistics.time_series import construct

import numpy as np

def online_changepoint_detection(data, hazard_func, observation_likelihood, threshold):
    n = len(data)
    R = np.zeros((n + 1, n + 1))
    R[0, 0] = 1
    S = np.zeros((n + 1, n + 1))  # Modified shape of S

    for t in range(1, n + 1):
        pred_probs = observation_likelihood(data[t - 1], np.arange(t), threshold)
        pred_probs = np.pad(pred_probs, (1, 1), 'constant')

        growth_probs = np.ones((2, 2), dtype=float)
        pcp = np.sum(growth_probs)

        if pcp == 0 or np.any(growth_probs == 0):
            S_new = np.copy(growth_probs)
        else:
            S_new = growth_probs / pcp

        S_new_flat = S_new.flatten()[:t + 1]
        S_temp = np.zeros((t + 2,))
        S_temp[1:t + 2] = S_new_flat
        S[t, :t + 1] = S_temp[:t + 1]  # Adjusted assignment of values to S

        R[0:t + 1, t] = (1 - pcp) * R[0:t + 1, t - 1]
        if t < n:
            R[t + 1, t] = pcp * hazard_func(t)

    R = R[1:]
    S = S[1:]

    changepoint_indices = []

    # Rest of the code...

    return changepoint_indices

def hazard_func(t, r=0.1):
    """
    Hazard function for the OCD algorithm.

    Args:
        t: Current time step.
        r: Changepoint rate.

    Returns:
        Hazard probability.
    """
    return r


def observation_likelihood(x, cp, threshold):
    """
    Observation likelihood function for the OCD algorithm with cusum-based approach.

    Args:
        x: Observation.
        cp: Changepoint positions.

    Returns:
        Likelihood probability.
    """
    if len(cp) > 0:
        x_slice = x[:cp[-1]]
        if len(x_slice) > 0:
            mean = np.mean(x_slice)
            cumulative_sum = np.zeros_like(x)
            drift_indices = []

            for i in range(1, len(x)):
                cumulative_sum[i] = np.maximum(0, cumulative_sum[i-1] + x[i] - mean - threshold)
                if np.any(cumulative_sum[i-1] > 0) and np.any(cumulative_sum[i] > 0):
                    drift_indices.append(i)

            likelihood = np.zeros(len(x))
            likelihood[drift_indices] = 1.0
        else:
            # Default likelihood value for empty slice
            likelihood = np.full(len(x), 1.0)
    else:
        # Default likelihood value for empty cp
        likelihood = np.full(len(x), 1.0)

    return likelihood


def main():
    logs = construct.subdivide_log(
        'pm4py/statistics/time_series/experiments/data/Synthetic_Insurance_Claim_large.xes',
        datetime(1970, 1, 15), datetime(1970, 10, 27), 1)

    primary_names, primary_features = construct.apply_feature_extraction(
        logs, ["direct_follows_relations"])
    reduced_primary = construct.pca_reduction(primary_features, 'mle', normalize=True, normalize_function="max")

    cp_1_indices = online_changepoint_detection(reduced_primary, hazard_func, observation_likelihood, threshold=0.9)

    print("Changepoint Indices:", cp_1_indices)


if __name__ == "__main__":
    main()
