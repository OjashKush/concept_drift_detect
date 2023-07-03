from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from pm4py.statistics.time_series import  construct 
from pm4py.statistics.time_series import  change_points 
from pm4py.statistics.time_series import  cause_effect
def main():
    
    logs = construct.subdivide_log(
        'pm4py/statistics/time_series/experiments/data/Synthetic_Insurance_Claim_large.xes',
        datetime(1970,1,15), datetime(1970,10,27), 1)
    
    pen_primary = 3
    pen_secondary = 1.5
    
    primary_names, primary_features = construct.apply_feature_extraction(
        logs, ["duration"])
    #dimensionality reduction with automated choice of dimensionality
    reduced_primary = construct.pca_reduction(primary_features,'mle',
                                                             normalize = True, 
                                                             normalize_function="max")
    cp_1 = change_points.rpt_pelt(reduced_primary, pen = pen_primary)
    print(cp_1)
    
    secondary_names, secondary_features = construct.apply_feature_extraction(logs,
                                                    ["all_numeric_data"])
    #dimensionality reduction with automated choice of dimensionality
    reduced_secondary = construct.pca_reduction(secondary_features,
                                                               'mle', 
                                                               normalize = True,
                                                               normalize_function="max")
    cp_2 = change_points.rpt_pelt(reduced_secondary, pen = pen_secondary)
    print(cp_2)
    
    res = cause_effect.granger_causality(primary_features, secondary_features,
                                        secondary_names, cp_1, cp_2,
                                        p_value = 0.0000000000001)
    
    cause_effect.draw_ca(res, primary_features, primary_names, secondary_features, secondary_names, store_path="synthetic.pdf")

if __name__ == "__main__":
    # execute only if run as a script
    main()