import pandas as pd
import re
import pprint as pp

df = pd.read_excel('PPB_gang_records_UPDATED_100516.xlsx')

pd.read_excel('PPB_gang_records_UPDATED_100516.xlsx',sheetname=1).head()

metadata = pd.read_excel('PPB_gang_records_UPDATED_100516.xlsx',header=3,sheetname=1)

metadata_dict = dict(zip(metadata['Variable'], metadata['Explanation']))

pp.pprint(metadata_dict)

df = pd.read_excel('PPB_gang_records_UPDATED_100516.xlsx')

df.head()

df.info()

df['zip'].replace(regex=True,inplace=True,to_replace=r'\D',value=r'')

df.describe()

# Find the index of this maximum 'yob' (year of birth) row
df['yob'].idxmax(axis=1)

df.ix[306]['yob']

# Replace year of birth according to correction we received from the Portland Police Bureau.
df.set_value(306, 'yob',1998)

# Establish a new column for person's age at designation
df['Approximate Age at Designation'] = df['membership_date'].apply(lambda x: x.year)-df['yob']

df.head()

# Rather than count individual criteria in the "crime or conspiracy" subset,
# return `True` if any criteria therein are met

df['Any Crime'] = (df['crt3_checked'] == "Y") | (df['crt4_checked'] == "Y") | (df['crt5_checked'] == "Y") | (df['crt6_checked'] == "Y")|(df['crt7_checked'] == "Y") 

# This is the subset of criteria not related to "crime or conspiracy to
# commit a crime," e.g. clothing, tattoos, etc.

df_criteria = df[['crt1_checked','crt2_checked','crt8_checked','crt9_checked','crt10_checked','crt12_checked','crt13_checked','crt14_checked']]

# sum non-crime criteria met
def y_count(dataframe):
    counter = 0
    for x in dataframe:
        if x == "Y":
            counter+=1
    return counter

# perform count on dataframe composed only of non-crime criteria
# but append that information to original dataset
df['Non-Crime Criteria Met'] = df_criteria.apply(lambda x: y_count(x),axis=1)

df

# sum non-crime criteria and crime criteria, but counting
# the latter as one point, as `True` evaluates to 1.

df['Total Criteria Met'] = df['Any Crime'] + df['Non-Crime Criteria Met']

# Make human-readable labels for data, based on metadata sheet

rename_dict = {
"gang_name":"Gang Name",
"crt1_checked":"Claims Affiliation",
"crt2_checked":"Participated in Initiation",
"crt3_checked":"Crime: Gang-Assisted Self-Benefiting",
"crt4_checked":"Crime: Gang-Oriented Self-Benefiting",
"crt5_checked":"Crime: Gang-Benefiting",
"crt6_checked":"Crime: Gang-Promoting",
"crt7_checked":"Crime: Victim-Oriented",
"crt8_checked":"Knowledge of Gang Culture",
"crt9_checked":"Announces Allegiance",
"crt10_checked":"Gang Clothing or Jewelry",
"crt11_checked":"Gang Language",
"crt12_checked":"Named in a Gang Document",
"crt13_checked":"Appears in a Gang-Related Photograph",
"crt14_checked":"Gang Tattoo"}

df.rename(index=str, columns=rename_dict,inplace=True)

df.head()

df.to_csv('gang_list_for_visualization.csv')

