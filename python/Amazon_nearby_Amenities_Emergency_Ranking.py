# Dependencies
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

hospital = pd.read_csv('../Results/Hospital_Rating.csv')
del hospital['Unnamed: 0']
hospital.replace('NAN', value=0, inplace=True)
hospital = hospital.rename(columns={'hospital Total Count':'Total Count', 'Facility hospital':'Hospital Facility'})
hospital['Rating']=hospital['Rating'].astype(float)
hospital['Total Count']=hospital['Total Count'].astype(int)
hospital.head()

new_hospital = hospital.groupby(['City Name', 'Site Name'])
hospital_count_df = pd.DataFrame(new_hospital['Site Name'].value_counts())
hospital_count_df = hospital_count_df.rename(columns={'Site Name': 'Total Count'})
hospital_count_df = hospital_count_df.reset_index(level=1)
hospital_count_df = hospital_count_df.reset_index(level=0)
hospital_count_df = hospital_count_df.reset_index(drop=True)
hospital_count_df.head()

hospital_count_final = hospital_count_df.groupby(['City Name'])
hospital_count_final_df = pd.DataFrame(hospital_count_final['Total Count'].median())
hospital_count_final_df = hospital_count_final_df.sort_values(['Total Count'])[::-1]
hospital_count_final_df = hospital_count_final_df.reset_index()
hospital_count_final_df['Type']='Hospital'
hospital_count_final_df

firestation = pd.read_csv('../Results/FireStation_Rating.csv')
del firestation['Unnamed: 0']
firestation.replace('NAN', value=0, inplace=True)
firestation = firestation.rename(columns={'fire_station Total Count':'Total Count', 'Facility fire_station':'FireStation Facility'})
firestation['Rating']=firestation['Rating'].astype(float)
firestation['Total Count']=firestation['Total Count'].astype(int)
firestation.head()

new_firestation = firestation.groupby(['City Name', 'Site Name'])
firestation_count_df = pd.DataFrame(new_firestation['Site Name'].value_counts())
firestation_count_df = firestation_count_df.rename(columns={'Site Name': 'Total Count'})
firestation_count_df = firestation_count_df.reset_index(level=1)
firestation_count_df = firestation_count_df.reset_index(level=0)
firestation_count_df = firestation_count_df.reset_index(drop=True)
firestation_count_df.head()

firestation_count_final = firestation_count_df.groupby(['City Name'])
firestation_count_final_df = pd.DataFrame(firestation_count_final['Total Count'].median())
firestation_count_final_df = firestation_count_final_df.sort_values(['Total Count'])[::-1]
firestation_count_final_df = firestation_count_final_df.reset_index()
firestation_count_final_df['Type']='Fire Station'
firestation_count_final_df

doctor = pd.read_csv('../Results/Doctor_Rating.csv')
del doctor['Unnamed: 0']
doctor.replace('NAN', value=0, inplace=True)
doctor = doctor.rename(columns={'doctor Total Count':'Total Count', 'Facility doctor':'Doctor Facility'})
doctor['Rating']=doctor['Rating'].astype(float)
doctor['Total Count']=doctor['Total Count'].astype(int)
doctor.head()

new_doctor = doctor.groupby(['City Name', 'Site Name'])
doctor_count_df = pd.DataFrame(new_doctor['Site Name'].value_counts())
doctor_count_df = doctor_count_df.rename(columns={'Site Name': 'Total Count'})
doctor_count_df = doctor_count_df.reset_index(level=1)
doctor_count_df = doctor_count_df.reset_index(level=0)
doctor_count_df = doctor_count_df.reset_index(drop=True)
doctor_count_df.head()

doctor_count_final = doctor_count_df.groupby(['City Name'])
doctor_count_final_df = pd.DataFrame(doctor_count_final['Total Count'].median())
doctor_count_final_df = doctor_count_final_df.sort_values(['Total Count'])[::-1]
doctor_count_final_df = doctor_count_final_df.reset_index()
doctor_count_final_df['Type']='Doctor'
doctor_count_final_df

new_emergency_df = hospital_count_final_df.append(firestation_count_final_df)
new_all_emergency_df = new_emergency_df.append(doctor_count_final_df)
new_all_emergency_df = new_all_emergency_df.reset_index(drop=True)
print(len(new_all_emergency_df))
new_all_emergency_df = new_all_emergency_df.drop([8,14,26])
print(len(new_all_emergency_df))
new_all_emergency_df = new_all_emergency_df.reset_index(drop=True)
new_all_emergency_df

print("========================================")
print("==================TEST====================")

sns.factorplot(kind='bar',x='Type',y='Total Count',data=new_all_emergency_df,
               hue='City Name', size=5, aspect=2.5)

total_count = new_all_emergency_df.groupby(['City Name'])['Total Count'].median().sort_values()[::-1].reset_index()
total_count_df = pd.DataFrame(total_count)
print(total_count_df)
ranks_dict = {}
y=1
for name in total_count_df['City Name']:
    ranks_dict[name] = y
    y=y+1
print(ranks_dict)

plt.title('City Emergency Ranking', fontsize=20, fontweight='bold')

plt.xlabel(' ', fontsize=15)
plt.ylabel('Median Count', fontsize=15)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

new_labels = ['#1 New York', '#2 Chicago', '#3 Boston', '#4 Washington DC', '#5 Los Angeles', '#6 Atlanta',
              '#7 Raleigh', '#8 Austin']
plt.legend(new_labels, frameon=False, title='Rank',
           bbox_to_anchor=(.365, 1), loc=1, borderaxespad=0.)


print("========================================")
print("==================END====================")
plt.savefig('Save_Figs/Emergency.png', bbox_inches='tight')

plt.show()

