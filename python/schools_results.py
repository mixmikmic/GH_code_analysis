import requests as req
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os


mainfile = os.path.join("..","Project1_AmazonSites.xlsx")
# read sites 
xls = pd.ExcelFile(mainfile) 
sites_df=xls.parse('AmazonSites', dtype=str) 
sites_df.head()

# read data from Great Schools!

schoolsfile = os.path.join("Results","schools_GreatSchool.csv")
school_sites= pd.read_csv(schoolsfile)

#reduce to cities
cities = sites_df[['Amazon City', 'Site Name']]
cities = cities.merge(school_sites)

cities = cities.sort_values('Average School Rating', ascending=False)
sns.set_style("whitegrid")
plt.figure(figsize=(10,5))
sns.boxplot(x='Average School Rating',y='Amazon City',data=cities)
plt.title("Average rating of nearby schools by City")
sch_rating_file = os.path.join("Plots","ratingsschoolscity.png")
plt.savefig(sch_rating_file)
plt.show()

ranking_schools=cities.groupby(['Amazon City'])['Average School Rating'].mean()
ranking_schools=ranking_schools.reset_index()
ranking_schools = ranking_schools.sort_values('Average School Rating', ascending=False)
ranking_schools = ranking_schools.rename(columns={'Amazon City':'City'})
ranking_schools = ranking_schools.drop([6])
ranking_schools = ranking_schools.drop(['Average School Rating'],axis=1)
ranking_schools

ordered = np.arange(8,0,-1)
ranking_schools['schools'] = ordered
ranking_schools = ranking_schools.set_index('City')
ranking_schools



ordered



collegefile = os.path.join("Results","Colleges.csv")
schools_df = pd.read_csv(collegefile)

schools_df['Rating'] = schools_df['Rating'].astype(float)
m = schools_df.groupby('Site Name')['Rating'].mean()
am = pd.DataFrame(m)
am = am.reset_index()
am = am.sort_values('Rating', ascending=False)
plt.figure(figsize=(5,10))
# sns.barplot(x='Rating',y='Site Name',data=am)
# plt.title("Average rating of nearby colleges")
# plt.savefig("nearbycollegerating.png")
# plt.show()

#reduce to cities
cities = sites_df[['Amazon City', 'Site Name']]
cities = cities.merge(am)
cities = cities.sort_values('Rating', ascending=False)
sns.set_style("whitegrid")
plt.figure(figsize=(10,5))
sns.boxplot(x='Rating',y='Amazon City',data=cities)
plt.title("Average rating of nearby colleges by City")
col_rating_file = os.path.join("Plots","collegeratingscity.png")
plt.savefig(col_rating_file)
plt.show()

ranking_college=cities.groupby('Amazon City').mean()
ranking_college = ranking_college.reset_index()
ranking_college = ranking_college.sort_values('Rating', ascending=False)
ranking_college = ranking_college.rename(columns={'Amazon City':'City'})
ranking_college

ranking_college = ranking_college.drop([6])
ranking_college = ranking_college.drop(['Rating'],axis=1)
ranking_college['college'] = ordered
ranking_college=ranking_college.set_index('City')
ranking_college

ranking_schools['college']=ranking_college['college']

ranking_schools

school_ranking_file = os.path.join("Results","school_ranking.csv")
ranking_schools.to_csv(school_ranking_file)



