import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon

#Read the csv file that was downloaded from the City of Austin: "https://data.austintexas.gov/"
Austin_311_df = pd.read_csv('data/All_Austin_311.csv',low_memory=False)

Austin_311_df.head()

#Drop the Type of Complaint, Complaint Description, and Date since they are not needed.
datetime_austin311 = Austin_311_df.drop(["complaint_description",
                                         "complaint_type", "incident_zip",
                                        "latitude", "longitude"], axis=1)
datetime_austin311.head()

#Pivot the table to make 'Year' the index and get the counts for Department 
# and change the name of Department to Number of Complaints.

datetime_austin311 = datetime_austin311.rename(columns={"owning_department":"Number of Complaints",
                                                     "year":"Year", "month":"Month"})
austin311_by_year = datetime_austin311.pivot_table(datetime_austin311, index=['Year','Month'], aggfunc='count')

austin311_by_year.head()

#Drop the year 2013 since it has only one month, December and 2107 since it only has 8 months, Jan - Aug.

austin311_by_year.drop(austin311_by_year.index[0],inplace=True)
austin311_by_year.drop(austin311_by_year.index[36:],inplace=True)
austin311_by_year

#Pivot the df so 'Month' is the index 

austin311_by_month = austin311_by_year.pivot_table('Number of Complaints', ['Month'],'Year' )

#Reset the index to and save to .cvs file.
austin311_reset = austin311_by_month.reset_index()

#Save the file as a .csv
austin311_reset.to_csv('data/austin311_year_month.csv', encoding='utf-8', index=False)

austin311_reset

# Plot the number of calls by year with the months on the x-axis. Find out what if the year is the index or not. 
years = austin311_by_month.keys()
years

# Plot the number of calls by year.
plt.figure(figsize=(10,10))

plot = austin311_by_month.plot(kind='line')
plt.xticks(austin311_by_month.index)

plt.xlabel("Months", fontsize=14)
plt.ylabel("Number of Calls", fontsize=14)
plt.title("Number of 311 Calls 2014-2016", fontsize=16)
plt.legend(bbox_to_anchor=(1.05,1),loc= 2, borderaxespad = 0.,title="Year")

fig1 = plt.gcf()
plt.show()
fig1.savefig('images/Number_311_calls.png', bbox_inches='tight', dpi=200)



