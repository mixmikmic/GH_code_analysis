import numpy as np
import pandas as pd

# Import data csv into dataframe
df = pd.read_csv('df_data_inspection_cleaning_output.csv')
df = df.drop('Unnamed: 0', axis = 1)

df.head()

df_2 = df.copy()

# Drop DOMP column for target = CloseDate_dt_month.
#df_2 = df_2.drop('DOMP',axis=1)

# Drop TotalTaxes2 column.
#df_2 = df_2.drop('TotalTaxes2',axis=1)

# Drop ListPrice2_delta column for target = ListPrice2.
#df_2 = df_2.drop('ListPrice2_delta',axis=1)

# Drop CloseDate and ClosePrice columns (i.e. in-conjunction with ListDate columns
# informs DOMP directly) to reduce data leakage for modelling. Also ListDate_dt_year,
# since the data is already subset for the year, combined with predicting days on market,
# regardless of year.
df_2 = df_2.drop('CloseDate_dt_year',axis=1)
df_2 = df_2.drop('CloseDate_dt_month',axis=1)
df_2 = df_2.drop('CloseDate_dt_day',axis=1)
#df_2 = df_2.drop('ClosePrice2',axis=1)
df_2 = df_2.drop('ListDate_dt_year',axis=1)
#df_2 = df_2.drop('ListDate_dt_month',axis=1)
#df_2 = df_2.drop('ListDate_dt_day',axis=1)

# Drop Longitude and Latitude columns since we have geographic categorization with Zipcode.
df_2 = df_2.drop('PropertyLatitude',axis=1)
df_2 = df_2.drop('PropertyLongitude',axis=1)

# Drop mimiStatus column since we have mimi number column.
df_2 = df_2.drop('mimiStatus',axis=1)

# Drop FreddieMac5yrARM column since we have FreddieMac15yr column.
df_2 = df_2.drop('FreddieMac5yrARM',axis=1)

# Drop specific demographic columns. These are heavily coorelated with the SchoolDigger
# educational data.
df_2 = df_2.drop('PctHshldCar_2010_14',axis=1)
df_2 = df_2.drop('PctHshldPhone_2010_14',axis=1)
df_2 = df_2.drop('PctFamiliesOwnChildrenFH_2010_14',axis=1)
df_2 = df_2.drop('PctPoorChildren_2010_14',axis=1)
df_2 = df_2.drop('PctPoorElderly_2010_14',axis=1)
df_2 = df_2.drop('Pct16andOverEmployed_2010_14',axis=1)
df_2 = df_2.drop('Pct25andOverWoutHS_2010_14',axis=1)
df_2 = df_2.drop('PctForeignBorn_2010_14',axis=1)
df_2 = df_2.drop('PctPoorPersons_2010_14',axis=1)
df_2 = df_2.drop('PctUnemployed_2010_14',axis=1)

# Drop specific SchoolDigger educational columns.
# Have the StarRating (scale 0-5)
df_2 = df_2.drop('ES_AvgStandardScore',axis=1)
df_2 = df_2.drop('HS_AvgStandardScore',axis=1)
df_2 = df_2.drop('MS_AvgStandardScore',axis=1)
df_2 = df_2.drop('ES_Rank',axis=1)
df_2 = df_2.drop('HS_Rank',axis=1)
df_2 = df_2.drop('MS_Rank',axis=1)
# Parallel coordinates plot indicated not important
df_2 = df_2.drop('ES_IsCharter',axis=1)
df_2 = df_2.drop('ES_IsMagnet',axis=1)
df_2 = df_2.drop('ES_IsVirtual',axis=1)
df_2 = df_2.drop('ES_IsTitleI',axis=1)
df_2 = df_2.drop('HS_IsCharter',axis=1)
df_2 = df_2.drop('HS_IsMagnet',axis=1)
df_2 = df_2.drop('HS_IsVirtual',axis=1)
df_2 = df_2.drop('HS_IsTitleI',axis=1)
df_2 = df_2.drop('MS_IsCharter',axis=1)
df_2 = df_2.drop('MS_IsMagnet',axis=1)
df_2 = df_2.drop('MS_IsVirtual',axis=1)
df_2 = df_2.drop('MS_IsTitleI',axis=1)
# Have student-teacher ratio
df_2 = df_2.drop('ES_NumFTTeachers',axis=1)
df_2 = df_2.drop('ES_NumStudents',axis=1)
df_2 = df_2.drop('HS_NumFTTeachers',axis=1)
df_2 = df_2.drop('HS_NumStudents',axis=1)
df_2 = df_2.drop('MS_NumFTTeachers',axis=1)
df_2 = df_2.drop('MS_NumStudents',axis=1)

# Drop count columns since we have the distance columns. Keeping Grocery and Metro columns
# due to greater amount of consumer choice, compared to schools.
df_2 = df_2.drop('count_public_school_arts_center_km',axis=1)
df_2 = df_2.drop('count_cap_gain_school_km',axis=1)
#df_2 = df_2.drop('count_grocery_km',axis=1)
df_2 = df_2.drop('count_ind_school_km',axis=1)
#df_2 = df_2.drop('count_metro_bus_km',axis=1) 
#df_2 = df_2.drop('count_metro_station_km',axis=1)
df_2 = df_2.drop('count_public_school_edu_campus_km',axis=1)
df_2 = df_2.drop('count_public_school_elem_km',axis=1)
df_2 = df_2.drop('count_public_school_elem_specialized_km',axis=1)
df_2 = df_2.drop('count_public_school_high_km',axis=1)
df_2 = df_2.drop('count_public_school_high_specialized_km',axis=1)
df_2 = df_2.drop('count_public_school_mid_km',axis=1)
df_2 = df_2.drop('count_public_school_special_ed_km',axis=1)
df_2 = df_2.drop('count_public_school_ye_km',axis=1)

# Get dummies for specific features
df_zc = pd.get_dummies(df_2['Zip'],prefix='zip')
df_2 = pd.concat([df_2,df_zc], axis=1)
df_2 = df_2.drop('Zip', axis=1)

df_ltm = pd.get_dummies(df_2['ListDate_dt_month'],prefix='ldmonth')
df_2 = pd.concat([df_2,df_ltm], axis=1)
df_2 = df_2.drop('ListDate_dt_month', axis=1)

df_ltd = pd.get_dummies(df_2['ListDate_dt_day'],prefix='ldday')
df_2 = pd.concat([df_2,df_ltd], axis=1)
df_2 = df_2.drop('ListDate_dt_day', axis=1)

df_essr = pd.get_dummies(df_2['ES_SDStarRating'],prefix='ESSR')
df_2 = pd.concat([df_2,df_essr], axis=1)
df_2 = df_2.drop('ES_SDStarRating', axis=1)

df_hssr = pd.get_dummies(df_2['HS_SDStarRating'],prefix='HSSR')
df_2 = pd.concat([df_2,df_hssr], axis=1)
df_2 = df_2.drop('HS_SDStarRating', axis=1)

df_mssr = pd.get_dummies(df_2['MS_SDStarRating'],prefix='MSSR')
df_2 = pd.concat([df_2,df_mssr], axis=1)
df_2 = df_2.drop('MS_SDStarRating', axis=1)

# Convert mimiStatus back to str values and then get dummies
# df_2['mimiStatus'] = df_2['mimiStatus'].replace([0,1,2],['Weak','Elevated','In Range'], inplace=False)

# df_ms = pd.get_dummies(df_2['mimiStatus'],prefix='mimiStatus')
# df_2 = pd.concat([df_2,df_ms], axis=1)
# df_2 = df_2.drop('mimiStatus', axis=1)

df_2.head()

df_2.columns

df_2.describe()

# Export dataframe to disk
df_2.to_csv('df_prep_for_feature_selection_output.csv')



