# Imports
import pandas as pd

# Need the source files for FraudHacker.
import sys
sys.path.append('/home/dan/PycharmProjects/fraudhacker/src')

from anomaly_tools import HDBAnomalyDetector
from database_tools import PandasDBReader
from fh_config import regression_vars, response_var

# Some routines to make it a bit simpler.
def get_df_with_counts(states, specialty):
    pdb_reader = PandasDBReader("./config.yaml", states, specialty)
    hdb = HDBAnomalyDetector(regression_vars, response_var, pdb_reader.d_f, use_response_var=True)
    hdb.get_outlier_scores(min_size=15)
    return hdb.get_most_frequent()

def build_new_table_data(states, specialty):
    counted_df = get_df_with_counts(states, specialty)
    new_table_data = {
        'state': [ginfo['state'] for ginfo in counted_df['address'].values],
        'lastname': counted_df['last_name'],
        'provider_type': [specialty[0] for i in range(len(list(counted_df.index)))],
        'outlier_count': counted_df['outlier_count'],
        'outlier_rate': counted_df['outlier_count_rate'],
        'cost': counted_df['cost_to_medicare'],
    }
    return pd.DataFrame(data=new_table_data, index=list(counted_df.index))

# Temporarily testing some of the new functionality.
test_state = "TX"
test_spec = "Internal Medicine"
table_data = build_new_table_data([test_state], [test_spec])

print(table_data.loc[table_data["lastname"] == "LA HOZ"])
print(table_data.loc[table_data["lastname"] == "MASTERSON"])

csv_output = "/home/dan/insight/cms_claims/master_outlier_counts_more_data_fixed.csv"
states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA", 
          "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", 
          "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 
          "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", 
          "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]
specialties = ['Internal Medicine', 'Family Practice', 'Psychiatry', 
               'Neurology', 'Endocrinology', 'Physical Medicine and Rehabilitation']
for state in states:
    for specialty in specialties:
        table_data = build_new_table_data([state], [specialty])
        with open(csv_output, 'a') as f:
            table_data.to_csv(f)
        print("Finished " + specialty + " in " + state + ".")



