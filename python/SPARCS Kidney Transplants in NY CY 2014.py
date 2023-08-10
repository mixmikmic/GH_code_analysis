import pandas as pd

kt_url = 'https://health.data.ny.gov/resource/rmwa-zns4.json?ccs_procedure_code=105&$limit=10000'
print(kt_url)

kidney_transplants_df = pd.read_json(kt_url)

len(kidney_transplants_df.length_of_stay)

kidney_transplants_df.head(10)

#Uncomment last line if having issues with the live API connection
#kidney_transplants_df.to_csv("./data/sparcs_ny_kidney_transplants_2014.csv")
#kidney_transpants_df = pd.read_csv("./data/sparcs_ny_kidney_transplants_2014.csv")

kidney_transplants_df.columns

get_ipython().magic('matplotlib inline')

import seaborn as sb

sb.violinplot(x="age_group", y="length_of_stay", hue="gender", data=kidney_transplants_df)

sb.violinplot(x="facility_id", y="length_of_stay", data=kidney_transplants_df)

sb.violinplot(x="facility_id", y="length_of_stay", data=kidney_transplants_df)

kidney_transplants_outliers_removed_df = kidney_transplants_df.where(kidney_transplants_df["length_of_stay"] <= 40)

sb.violinplot(x="facility_id", y="length_of_stay", data=kidney_transplants_outliers_removed_df)

kidney_transplants_df["facility_name_with_id"] = kidney_transplants_df.apply(
    lambda x: str("000" + str(int(x["facility_id"])))[-4:] + " - " + x["facility_name"], axis=1)

kidney_transplants_df.groupby(["facility_name_with_id"])["length_of_stay"].mean()

kidney_transplants_df.groupby(["facility_name_with_id"])["length_of_stay"].count()

kidney_transplants_df.groupby(["patient_disposition", "facility_name"])["length_of_stay"].count()

kidney_transplants_df.groupby(["age_group"])["length_of_stay"].count()

kidney_transplants_df.groupby(["age_group"])["length_of_stay"].mean()

kidney_transplants_df.groupby(["patient_disposition"])["length_of_stay"].count()

sb.countplot(x="admit_day_of_week", data=kidney_transplants_df)

sb.countplot(x="admit_day_of_week", data=kidney_transplants_df, order=["SUN","MON", "TUE", "WED", "THU", "FRI", "SAT"])

