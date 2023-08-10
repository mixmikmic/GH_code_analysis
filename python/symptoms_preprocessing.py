import pandas as pd
import numpy as np
import os
import sys
import zeep
from zeep import Client
from lxml import etree
from zeep import Plugin

#Read in data
df = pd.read_csv('symptoms.csv', sep=',',encoding='latin-1')

df.reset_index(drop=True)

#Drop unecessary columns
df = df.drop(columns=['Unnamed: 0','[PathogenMitKrankheit].[Meldeweg Web71].[Meldeweg ID Web71].[MEMBER_CAPTION]'])

#Give columns more comprehensible names
df = df.rename(columns={'[PathogenMitKrankheit].[Meldeweg Web71].[Krankheit ID Web71].[MEMBER_CAPTION]': 'Krankheit', '[Symptome].[ID].[ID].[MEMBER_CAPTION]': 'Symptom', '[ReportingDate].[YearWeek].[YearWeek].[MEMBER_CAPTION]': 'Kalenderwoche', '[Measures].[FallCount]': 'Anzahl'}) 

#Since we want to focus only on Norovirus, Infulenza, and Windpocken, we throw all the other data away
df_selection = df[df.isin({'Krankheit': ['Norovirus-Gastroenteritis','Influenza', 'Windpocken']})['Krankheit']]

#df_selection_symptoms = df_selection.groupby(['Krankheit','Symptom'])['Anzahl'].sum().reset_index()
#df_selection_symptoms

#Further dropping
df_selection = df_selection.drop(columns='Krankheit')

df_selection_pivot = df_selection.groupby(['Symptom','Kalenderwoche'])['Anzahl'].sum().reset_index()
df_selection_pivot.sort_values('Kalenderwoche')

#get a list of all relevant symptoms
df_selection_all_symptoms = df_selection.groupby(['Symptom'])['Anzahl'].sum().reset_index()['Symptom'].tolist()

#for each combination of symptom and week:
#if the combination is not in the dataframe, then insert a row with this combination and 'Anzahl' = 0
new_rows = []
for week in df['Kalenderwoche'].drop_duplicates().sort_values():
    for symptom in df_selection_all_symptoms:
        sample_week = df_selection_pivot[df_selection_pivot['Kalenderwoche'] == week]
        if not sample_week.isin({'Symptom': [symptom]})['Symptom'].any():
            new_row = [symptom, week, 0]
            new_rows.append(new_row)
            
            
new_values = pd.DataFrame(new_rows, columns=['Symptom','Kalenderwoche','Anzahl'])

#append these illnesses with 0 'Anzahl'
df_selection_pivot= df_selection_pivot.append(new_values)
df_weekly_symptoms = df_selection_pivot.sort_values('Kalenderwoche').reset_index(drop=True)
df_weekly_symptoms = df_weekly_symptoms.sort_values(['Kalenderwoche','Symptom'],ascending=[True,True])

# Reset the index for ease of further processing
df_weekly_symptoms = df_weekly_symptoms.reset_index(drop=True)
print(df_weekly_symptoms.head(20))

df_weekly_symptoms.to_pickle('weekly_symptoms.pkl')

