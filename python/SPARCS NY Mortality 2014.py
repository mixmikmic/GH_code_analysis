import pandas as pd

base_url = "https://health.data.ny.gov/resource/rmwa-zns4"

pd.read_json(base_url + "?" + "$select=count(*)")

url_count_by_disposition = base_url + "?" + "$select=count(*),patient_disposition&$group=patient_disposition"
print(url_count_by_disposition)
pd.read_json(url_count_by_disposition)

url_facility_discharge_count = base_url + "?$select=count(*),facility_name&$group=facility_name"
print(url_facility_discharge_count)
facility_discharge_count_df = pd.read_json(url_facility_discharge_count)
facility_discharge_count_df.columns = ["discharge_count", "facility_name"]

facility_discharge_count_df.head(5)

url_facility_expired_count = base_url + "?$select=count(*),facility_name&$group=facility_name&patient_disposition=Expired"
print(url_facility_expired_count)
facility_expired_count_df = pd.read_json(url_facility_expired_count)
facility_expired_count_df.columns = ["expired_count", "facility_name"]

facility_expired_count_df.head(5)

facility_discharged_with_expired_df = pd.merge(facility_discharge_count_df, facility_expired_count_df, on="facility_name")

facility_discharged_with_expired_df["expired_rate_per_discharge"] = facility_discharged_with_expired_df["expired_count"] / facility_discharged_with_expired_df["discharge_count"]

facility_discharged_with_expired_df.sort_values("expired_rate_per_discharge", ascending=False).head(25)

facility_discharged_with_expired_df.sort_values("expired_rate_per_discharge", ascending=False)[facility_discharged_with_expired_df["discharge_count"] >= 1000]

hospital_url = base_url + "?facility_name=Coney%20Island%20Hospital&$limit=50000"
print(hospital_url)
hospital_df = pd.read_json(hospital_url)

# Uncomment the last line if there are API issues
#hospital_df.to_csv("./data/sparcs_hospital_apr_drg_2014.csv")
#hospital_df = pd.read_csv("./data/sparcs_hospital_apr_drg_2014.csv")

hospital_df.head(10)

hospital_df["apr_drg_with_code"] = hospital_df.apply(
    lambda x: str("00" + str(int(x["apr_drg_code"])))[-3:] + " - " + x["apr_drg_description"], axis=1)

hospital_df["length_of_stay"] = hospital_df.apply(
    lambda x: 120 if "120 +" == x["length_of_stay"] else int(x["length_of_stay"]), axis=1)

hospital_df["length_of_stay"].count()

import numpy as np
apr_drgs_df = hospital_df.groupby("apr_drg_with_code")["length_of_stay"].agg({"discharges": np.size, "total_days": np.sum, 
                                                                "mean_length_of_stay": np.mean})

apr_drgs_expired_df=hospital_df[hospital_df["patient_disposition"] == "Expired"].groupby("apr_drg_with_code")["length_of_stay"].agg({"number_of_in_hospital_deaths": np.size})

merged_apr_drgs_df = pd.merge(apr_drgs_df.reset_index(level=["apr_drg_with_code"]),apr_drgs_expired_df.reset_index(level=["apr_drg_with_code"]), on="apr_drg_with_code")

merged_apr_drgs_df.head(5)

merged_apr_drgs_df.apr_drg_with_code.count()

merged_apr_drgs_df["number_hospital_deaths_per_discharge"] = merged_apr_drgs_df["number_of_in_hospital_deaths"] / merged_apr_drgs_df["discharges"]

merged_apr_drgs_df["number_hospital_deaths_per_patient_days"] = merged_apr_drgs_df["number_of_in_hospital_deaths"] / merged_apr_drgs_df["total_days"]

merged_apr_drgs_df.sort_values("number_hospital_deaths_per_discharge", ascending=False)[merged_apr_drgs_df["discharges"] >= 10]

