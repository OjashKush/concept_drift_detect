from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from pm4py.statistics.time_series import construct
from pm4py.statistics.time_series import change_points
from pm4py.statistics.time_series import cause_effect

def main():
    logs = construct.subdivide_log(
        'pm4py/statistics/time_series/experiments/data/BPI2017.xes',
        datetime(2016, 1, 1), datetime(2016, 12, 24), 7)

    pen_primary = 3
    pen_secondary = 1.5

    primary_names, primary_features = construct.apply_feature_extraction(
        logs, ["all_numeric_data"])
    reduced_primary = construct.pca_reduction(primary_features, 'mle',
                                              normalize=True,
                                              normalize_function="max")
    cp_1 = change_points.rpt_pelt(reduced_primary, pen=pen_primary)
    print(cp_1)

    secondary_names, secondary_features = construct.apply_feature_extraction(logs,
                                                                             ["workload"])
    reduced_secondary = construct.pca_reduction(secondary_features,
                                                'mle',
                                                normalize=True,
                                                normalize_function="max")
    cp_2 = change_points.rpt_pelt(reduced_secondary, pen=pen_secondary)
    print(cp_2)


if __name__ == "__main__":
    # execute only if run as a script
    main()
