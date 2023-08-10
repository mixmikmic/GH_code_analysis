import pandas as pd

year_to_sparcs_key = {2009: "q6hk-esrj", 2010: "mtfm-rxf4", 2011: "pyhr-5eas", 2012: "u4ud-w55t",
                      2013: "npsr-cm47", 2014: "rmwa-zns4"}

base_url = "https://health.data.ny.gov/resource/"

def get_df_across_multiple_files(key_dict, base_url, search_query, row_limit=10000):
    df_dict = {} # Store df as keys
    for key in key_dict:
        ds_hash = key_dict[key]
        request_url = base_url + ds_hash + ".json" + "?" + search_query + "&$limit=" + str(row_limit)
        print("Extracting %s with %s" % (key, request_url))
        df = pd.read_json(request_url)
        df_dict[key] = df
    return df_dict

kd_df_dicts = get_df_across_multiple_files(year_to_sparcs_key, base_url, "ccs_procedure_code=105")

def create_single_df_from_df_dict(df_dict, key_name):
    """Build a single dataframe that that has a new field called key_name with the key values in df_dict"""
    df_keys = df_dict.keys()
    base_df = df_dict[df_keys[0]]
    base_df[key_name] = df_keys[0]
    for df_key in df_keys[1:]:
        df = df_dict[df_key]
        df[key_name] = df_key
        base_df = base_df.append(df, ignore_index=True)
    return base_df

kidney_cy_09_14 = create_single_df_from_df_dict(kd_df_dicts, "discharge_year")

# Uncomment the last line if there are API issues
#kidney_cy_09_14.to_csv("./data/sparcs_ny_kidney_transplants_2009_2014.csv")
#kidney_cy_09_14 = pd.read_csv("./data/sparcs_ny_kidney_transplants_2009_2014.csv")

kidney_cy_09_14.discharge_year.count()

kidney_cy_09_14.groupby("discharge_year")["length_of_stay"].count()

kidney_cy_09_14["length_of_stay"] = pd.to_numeric(kidney_cy_09_14["length_of_stay"], errors='coerce')

kidney_cy_09_14.groupby(["facility_name"])["length_of_stay"].mean()

pd.crosstab(kidney_cy_09_14["facility_name"],  kidney_cy_09_14["discharge_year"], margins=True)

import numpy as np

pd.crosstab(kidney_cy_09_14["facility_name"],  kidney_cy_09_14["discharge_year"], 
            margins=True, values=kidney_cy_09_14["length_of_stay"],aggfunc=np.mean)

pd.crosstab(kidney_cy_09_14["facility_name"],  kidney_cy_09_14["discharge_year"], 
            margins=True, values=kidney_cy_09_14["length_of_stay"], aggfunc=np.median)

import seaborn as sb

get_ipython().magic('matplotlib inline')

sb.boxplot(y="length_of_stay", x="discharge_year", data=kidney_cy_09_14)

sb.boxplot(y="length_of_stay", x="discharge_year", data=kidney_cy_09_14[kidney_cy_09_14["length_of_stay"] <= 20])

sb.violinplot(y="length_of_stay", x="discharge_year", data=kidney_cy_09_14[kidney_cy_09_14["length_of_stay"] <= 20])

