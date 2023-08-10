# Dependencies
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

supermarket = pd.read_csv('../Results/SuperMarket_Rating.csv')
del supermarket['Unnamed: 0']
supermarket.replace('NAN', value=0, inplace=True)
supermarket = supermarket.rename(columns={'supermarket Total Count':'Total Count', 'Facility supermarket':'SuperMarket Facility'})
supermarket['Rating']=supermarket['Rating'].astype(float)
supermarket['Total Count']=supermarket['Total Count'].astype(int)
supermarket.head()

new_supermarket = supermarket.groupby(['City Name', 'Site Name'])
supermarket_count_df = pd.DataFrame(new_supermarket['Site Name'].value_counts())
supermarket_count_df = supermarket_count_df.rename(columns={'Site Name': 'Total Count'})
supermarket_count_df = supermarket_count_df.reset_index(level=1)
supermarket_count_df = supermarket_count_df.reset_index(level=0)
supermarket_count_df = supermarket_count_df.reset_index(drop=True)
supermarket_count_df.head()

supermarket_count_final = supermarket_count_df.groupby(['City Name'])
supermarket_count_final_df = pd.DataFrame(supermarket_count_final['Total Count'].median())
supermarket_count_final_df = supermarket_count_final_df.sort_values(['Total Count'])[::-1]
supermarket_count_final_df = supermarket_count_final_df.reset_index()
supermarket_count_final_df['Type']='Supermarket'
supermarket_count_final_df

postoff = pd.read_csv('../Results/PostOffice_Rating.csv')
del postoff['Unnamed: 0']
postoff.replace('NAN', value=0, inplace=True)
postoff = postoff.rename(columns={'postoffice Total Count':'Total Count', 'Facility postoffice':'Post Office Facility'})
postoff['Rating']=postoff['Rating'].astype(float)
postoff['Total Count']=postoff['Total Count'].astype(int)
postoff.head()

new_postoff = postoff.groupby(['City Name', 'Site Name'])
postoff_count_df = pd.DataFrame(new_postoff['Site Name'].value_counts())
postoff_count_df = postoff_count_df.rename(columns={'Site Name': 'Total Count'})
postoff_count_df = postoff_count_df.reset_index(level=1)
postoff_count_df = postoff_count_df.reset_index(level=0)
postoff_count_df = postoff_count_df.reset_index(drop=True)
postoff_count_df.head()

postoff_count_final = postoff_count_df.groupby(['City Name'])
postoff_count_final_df = pd.DataFrame(postoff_count_final['Total Count'].median())
postoff_count_final_df = postoff_count_final_df.sort_values(['Total Count'])[::-1]
postoff_count_final_df = postoff_count_final_df.reset_index()
postoff_count_final_df['Type']='Post Office'
postoff_count_final_df

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

new_errands_df = supermarket_count_final_df.append(postoff_count_final_df)
new_all_errands_df = new_errands_df.append(doctor_count_final_df)
new_all_errands_df = new_all_errands_df.reset_index(drop=True)
print(len(new_all_errands_df))
new_all_errands_df
new_all_errands_df = new_all_errands_df.drop([4,13,26])
print(len(new_all_errands_df))
new_all_errands_df = new_all_errands_df.reset_index(drop=True)
new_all_errands_df

print("========================================")
print("==================TEST====================")

sns.factorplot(kind='bar',x='Type',y='Total Count',data=new_all_errands_df,
               hue='City Name', size=5, aspect=2.5)

total_count = new_all_errands_df.groupby(['City Name'])['Total Count'].median().sort_values()[::-1].reset_index()
total_count_df = pd.DataFrame(total_count)
print(total_count_df)
ranks_dict = {}
y=1
for name in total_count_df['City Name']:
    ranks_dict[name] = y
    y=y+1
print(ranks_dict)

plt.title('City Nearby Errands Ranking', fontsize=20, fontweight='bold')

plt.xlabel(' ', fontsize=15)
plt.ylabel('Median Count', fontsize=15)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

new_labels = ['#1 Boston', '#2 New York ', '#3 Chicago', '#4 Washington DC', '#5 Los Angeles', '#6 Atlanta',
              '#7 Austin', '#8 Raleigh']
plt.legend(new_labels, frameon=False, title='Rank',
           bbox_to_anchor=(.34, 1), loc=1, borderaxespad=0.)


print("========================================")
print("==================END====================")

plt.savefig('Save_Figs/Errands.png', bbox_inches='tight')

plt.show()

