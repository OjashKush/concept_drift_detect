import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from pm4py.statistics.time_series import construct


def online_changepoint_detection(data, hazard_func, observation_likelihood, threshold):
    n = len(data)
    R = np.zeros((n + 1, n + 1))
    R[0, 0] = 1
    S = np.zeros((n + 1, n))

    for t in range(1, n + 1):
        pred_probs = observation_likelihood(data[t - 1], np.arange(t), threshold)
        pred_probs = np.concatenate(([1], pred_probs[:-1]))

        growth_probs = R[1:t + 1, t - 1] * pred_probs[::-1]
        pcp = np.sum(growth_probs)

        R[0:t + 1, t] = (1 - pcp) * R[0:t + 1, t - 1]
        R[t + 1, t] = pcp * hazard_func(t)

        growth_probs_expanded = np.expand_dims(growth_probs, axis=0)
        S[1:t + 1, t - 1] = np.where(growth_probs_expanded != 0, growth_probs_expanded / pcp, 0)

    R = R[1:]
    S = S[1:]

    changepoint_indices = []
    i = n
    while i > 0:
        if S[i - 1, n - 2] > 0:
            changepoint_indices.append(i - 1)
            i = int(S[i - 1, n - 2] * (i - 1))
        else:
            i -= 1

    return changepoint_indices[::-1]





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
