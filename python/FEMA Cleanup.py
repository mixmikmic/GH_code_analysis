from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')

disaster_df = pd.read_csv('fema_data/DisasterDeclarationsSummaries_IL_Floods_Severe_Storms.csv')
print(disaster_df.dtypes)
print(disaster_df.shape)
disaster_df.head()

disaster_df['incidentBeginDate'] = pd.to_datetime(disaster_df['incidentBeginDate'])
disaster_recent = disaster_df.loc[disaster_df['incidentBeginDate'] >= '2000-01-01'].copy()
disaster_recent = disaster_recent.loc[disaster_recent['declaredCountyArea'] == 'Cook (County)'].copy()
disaster_recent.shape

disaster_recent

#disaster_recent.to_csv('fema_data/cook_disasters_since_2000.csv',index=False)
disaster_merge = disaster_recent[['disasterNumber', 'declarationDate', 'title', 'incidentBeginDate', 'incidentEndDate']]
disaster_merge

housing_owners = pd.read_csv('fema_data/HousingAssistanceOwners_IL_Cook_Flood_Storm.csv')
print(housing_owners.shape)
print(housing_owners.dtypes)
housing_owners.head()

housing_owners['city'].unique()

# There are some creative misspellings of Chicago, making a list of them and just pulling those rows
chi_names = ['CHIAGO', 'CHICAGI', 'CHGO', 'CHICAGO', 'CHICAGOI', 'CHICGAGO', 'CGICAGO', 
             'CHICAAGO', 'CHICAGIO', 'CHICAGOP', 'CHICGO', 'GHICAGO', 'CHICAGGO','CHICATO',
             'CHICAGO IL', 'CHICGAO', 'CHCIAGO', 'CHICAG', 'CHICAO', 'CHCAGO', 'CHICAHO',
             'CHHICAGO', 'CHICASGO', 'CHICACO', 'CHIC AGO', 'CHUCAGO', 'CHIHAGO']
chi_housing_owners = housing_owners.loc[housing_owners['city'].isin(chi_names)].copy()
chi_housing_owners['city'] = 'Chicago'
print(chi_housing_owners.shape)
chi_housing_owners.head()

chi_housing_recent = chi_housing_owners.merge(disaster_merge, on='disasterNumber', how='right')
print(chi_housing_recent.shape)
chi_housing_recent.head()

housing_renters = pd.read_csv('fema_data/HousingAssistanceRenters_IL_Cook_Flood_Storm.csv')
print(housing_renters.shape)
print(housing_renters.dtypes)
housing_renters.head()

housing_renters['city'].unique()

# Fix misspellings again
chi_names = ['CHCAGO', 'CHICAGO', 'CHICOAG', 'CHGO', 'CHICAO', 'CHICGO', 'CHCIAGO', 'CHICAGO APT B', 'CHICAAGO', 'CHICAGO ',
             'CHICAGOIL', 'CHICAGO IL']
chi_housing_renters = housing_renters.loc[housing_renters['city'].isin(chi_names)].copy()
chi_housing_renters['city'] = 'Chicago'
print(chi_housing_renters.shape)
chi_housing_renters.head()

chi_rent_recent = chi_housing_renters.merge(disaster_merge, on='disasterNumber', how='right')
print(chi_rent_recent.shape)
chi_rent_recent.head()

# chi_housing_recent.to_csv('chi_housing_assistance_owners.csv',index=False)
# chi_rent_recent.to_csv('chi_housing_assistance_renters.csv',index=False)

# Not sure if this dataset is relevant, ignoring for now
public_assistance = pd.read_csv('fema_data/PublicAssistanceApplicants_IL_Flood_Storm.csv')
print(public_assistance.shape)
print(public_assistance.dtypes)
public_assistance.head()

reg_data = pd.read_csv('fema_data/RegistrationIntakeIndividualsHouseholdPrograms_IL_Cook_Flood_Storm.csv')
print(reg_data.shape)
print(reg_data.dtypes)
reg_data.head()

reg_data['city'].unique()

# Cleaning up Chicago names again
# Fix misspellings again
chi_names = ['CHGO', 'CHICAGO', 'CHIHAGO', 'CHUCAGO', 'CHICGO', 'CGICAGO', 'CHICAG', 'CHHICAGO', 'CHCAGO', 'CHIAGO', 'CHCIAGO', 
             'CHICAGGO', 'CHICAGI', 'CHG', 'CHICAAGO', 'CHICAGIO', 'CHICACO', 'CHIC AGO', 'CHICAGO IL', 'CHICATO', 'CHICAGO ', 
             'CHICAGOP', 'CHICGAO', 'CHICAHO', 'CHICAGOI', 'CHICAO', 'CHICASGO', 'CHICOAG', 'CHICAGOIL', 'CHICGAGO', 'CHICAGO APT B']
chi_reg = reg_data.loc[reg_data['city'].isin(chi_names)].copy()
chi_reg['city'] = 'Chicago'
print(chi_reg.shape)
chi_reg.head()

chi_reg_recent = chi_reg.merge(disaster_merge, on='disasterNumber', how='right')
print(chi_reg_recent.shape)
chi_reg_recent.head()

#chi_reg_recent.to_csv('fema_data/chi_registration_intake.csv',index=False)

chi_reg_recent['zipCode'].unique()

reg_zip = chi_reg_recent.groupby(['zipCode'])['totalValidRegistrations'].sum()
reg_zip = reg_zip.sort_values(ascending=False)
reg_zip.plot(kind='bar')

reg_zip_df = pd.DataFrame(reg_zip).reset_index()
reg_zip_df.head()

reg_zip_df.to_csv('housing_assistance_reg_by_zip.csv',index=False)



