get_ipython().run_cell_magic('javascript', '', "$.getScript('https://kmahelona.github.io/ipython_notebook_goodies/ipython_notebook_toc.js')")

from collections import Counter
import pandas as pd
import numpy as np
import seaborn as sns
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

## 1. Read and understand the access data

df = pd.read_csv("/Users/Ben/Work/Vizzuality/SYNTHESIS/Data/FullExportAnon_v3.csv", encoding="mac_cyrillic")  # this dataset is private!

# Column names in dataframe
sorted(df.keys())

df.head()

def group_multiple_entries(df):
    """Provided a subset of a dataframe (which is intended only to be a subset where multiple entries 
    for the same applicant and project exist), we will condense down the data to a single row
    in order not to overweight such projects in the summary statistics.
    This will return a list tmp_row (list) which should be appended to a base_list. The keys will
    also be outputted (as they are not simply the same as the input dataframe).
    """
    tmp_row=[]    # hold each specific row, temporarily
    key_list =[]  # hold only a list of the keys we are going to pass, in the specific order of access
    for key in sorted(df.keys()):
        unique_element_entries = df[key].unique()
        if len(unique_element_entries) == 1:
            tmp_row.append(unique_element_entries[0])
            key_list.append(key)
        else:
            if key in ['NHM_Installation_Use.Installation_Long_Name',
                       'NHM_Installation_Use.Installation_Short_Name']:
                unique_element_entries = " &&& ".join(unique_element_entries) # encode info into a single string
                key_list.append(key)
                tmp_row.append(unique_element_entries)
                #print(f"key = {key}, entry = {unique_element_entries}")
            if key in ['NHM_Installation_Use.Amount_of_Access_Delivered']:
                unique_element_entries = unique_element_entries.sum()
                key_list.append(key)
                tmp_row.append(unique_element_entries)
                #print(f"key = {key}, entry = {unique_element_entries}")
            if key in ['NHM_Installation_Use.Installation_ID',
                       'HostInstName1',
                       'NHM_Installation_Use.Infrastructure_Short_Name']:
                string_ids = list(unique_element_entries)
                x_ids = [ str(ids) for ids in string_ids]
                x_ids = ' &&& '.join(x_ids)
                key_list.append(key)
                tmp_row.append(x_ids)
                #print(f"key = {key}, entry = {x_ids}")
    return tmp_row, key_list
    #new_base.append(tmp_row)  # After compressing each set of rows into a single row, add it to the new frame


#pd.DataFrame(new_base, columns=key_list)

print("Parsing dataset to remove multiple entries per user id and application id")
print("Adding ' &&& ' between all grouped strings")
print("Summing visit days")

## Look for cases where Application_ID UserProject_ID and HostInstName1 match
unique_applications = sorted(set(df['Application_ID'].values))

new_data = []

# For unique projects and applicants only have one row of data
for application in tqdm(unique_applications[0:]):  # <<----limit here
    applicant_mask = df['Application_ID']==application
    number_of_applications = len(df[applicant_mask])
    #print(f" Applicant: {application}")#, Num entries:{number_of_applications}")
    if number_of_applications > 1:
        applicant_df = df[applicant_mask]
        project_ids = applicant_df['UserProject_ID'].unique()
        #print(f"     {number_of_applications} applications over {len(project_ids)} projects ")
        for project_id in project_ids:
            #print(f"{application} {project_id}")
            mask_project_id = applicant_df['UserProject_ID'] == project_id
            df_same_applicant_and_project = applicant_df[mask_project_id]
            v, k = group_multiple_entries(df_same_applicant_and_project)
            #print(f"A: number of value entries: {len(v)}, number of key entries:{len(k)}")
            new_data.append(v)
        if len(k) < 31:
            print("Found bug!")
            break
    else:
        v, k = group_multiple_entries(df[applicant_mask])
        #print(f"B: number of value entries: {len(v)}, number of key entries:{len(k)}")
        new_data.append(v)
print("parsed dataset")
xdf = pd.DataFrame(new_data, columns=k)
print("created new dataframe")

print(f"Parsing reduced original dataframe by {len(df) - len(xdf):,g} entries")

df_raw = df # Save the old DF
df = xdf    # Overwrite the df keyword with the new (one row per project version)

print(f'Access dataset has {len(df):,g} rows')

unique_users = len(df['User_Code'].unique())
print(f"User_Code, the Anonymised user id column has {unique_users:,g} unqique entries")

d_users = Counter(df['User_Code']).most_common() # count and order the users

d_users[0:10]  # The top 10 users by number of appearances in the dataset

reccurence = [user_reccurence[1] for user_reccurence in d_users]

# Plot a simple histogram with binsize determined automatically
sns.distplot(reccurence, kde=False, color="b", axlabel='Number of visits')
plt.title("Reccuring users")
plt.ylabel("Number of user codes")
plt.tight_layout()

mask = df['User_Code'] == 'User1124'
df[mask].head()

# Example of the most frequently appearing users data:

def give_date(year, month, day):
    months = {'January': 1, 'Feburary': 2, 'March':3, 'April':4, 'May':5,
              'June':6, "July":7, "August":8,"September":9, "October":10,
              "November":11, "December":12}
    return pd.datetime(int(year), months['November'], int(day))



for index, row in df[mask].iterrows():
    date = give_date(row['ProjectStart_Year'], row['ProjectStart_Month'], row['ProjectStart_Day'])
    age = row['Applicant_Age_Visit_Start']
    to_use = row['NHM_Installation_Use.Installation_Long_Name']
    calls = row['Call_Submitted']
    print('Start:',date.date(), "age:",age, calls, to_use)

Counter(df['Call_Submitted']).most_common() # Not sure what this is exactly.

Counter(df['User_NHM.Home_Institution_Type']).most_common()

Counter(df['User_NHM.Researcher_status']).most_common()

Counter(df['Call_Submitted']).most_common()

Counter(df['NHM_Installation_Use.Amount_of_Access_Delivered']).most_common()[0:5] # Basically, amount of visit days funded.

Counter(df['NHM_Disciplines.DisciplineName']).most_common()  

Counter(df['NHM_Specific_Disciplines.SpecificDisciplineName']).most_common()[0:10]

Counter(df['NHM_Installation_Use.Installation_Long_Name']).most_common()

total_days = df['ProjectsView.length_of_visit'].sum()
print(f"{total_days:,g} research days at NHM installations.")

unique_users = len(df['User_Code'].unique())
print(f"{unique_users:,g} unique users")

unique_institutes = len(df['User_NHM.Home_Institution_Name'].unique())
print(f"Visitors from {unique_institutes:,g} different institutes")

unique_countries = len(df['User_NHM.Home_Institution_Country_code'].unique())
print(f"Visitors from {unique_countries:,g} countries")

def distribution(df, key, xlabel=None, ylabel=None, title=None):
    # Return a distribution plot e.g. to show the Age of users in a df object
    sns.distplot(df[key].values, kde=True, color="b", axlabel=xlabel)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()

# age of users for all collection
distribution(df, key='Applicant_Age_Visit_Start', ylabel="User age at start of project", xlabel='Age',
             title="Age of users (non-unique)")

def donut_plot(df, column_key, fix_keys=None, colors=None, json=False):
    """Pass an arbritrary dataframe with a key, and optional color scheme.
    Return a donut plot.
    """
    gender_counts = Counter(df[column_key])
    labels = []
    values = []
    if not fix_keys:
        for key in gender_counts:
            labels.append(key)
            values.append(gender_counts[key])
    else:
        for key in fix_keys:
            labels.append(key)
            values.append(gender_counts[key])        
    if json:
        return dict(gender_counts)
    else:
        explode = 0
        explode = (explode,) * len(labels)
        plt.pie(values, explode=explode, labels=labels,colors=colors,
                autopct='%1.1f%%', shadow=False)
        centre_circle = plt.Circle((0,0),0.75,color='black', fc='white',linewidth=0.75)
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)
        plt.axis('equal')
        plt.title("Gender balance")
        plt.show() 

# Gender balance of all users
donut_plot(df, column_key='User_NHM.Gender', fix_keys=["M","F"], colors=['lightskyblue', 'pink'])

# Can directly get the json instead like this 
donut_plot(df, column_key='User_NHM.Gender', json=True)



def visits_destination(df, otherize_threshold=None, json=False):
    """Pass in a dataframe and return a plot of visitors Installation destination"""
    place_list = Counter(df['NHM_Installation_Use.Installation_Long_Name']).most_common()
    if otherize_threshold:
        top_list_with_other=[]
        other_sum = 0
        percent_limit = int(len(df)*otherize_threshold)
        cumulative = 0
        for k,v in place_list:
            cumulative += v
            if cumulative < percent_limit:
                top_list_with_other.append([k,v])
            else:
                other_sum += v
        top_list_with_other.append(['other', other_sum])
        place_list = top_list_with_other
    if json:
        return dict(place_list)
    else:
        places = []
        visits = []
        for c, num in place_list:
            places.append(c)
            visits.append(num)
        place_visists = pd.DataFrame(place_list, columns=['Installation','visits'])
        f, ax = plt.subplots(figsize=(6, 10))
        # Plot the total crashes
        sns.set_color_codes("pastel")
        sns.barplot(x="visits", y="Installation", data=place_visists,
                    label="Total", color="r")
        ax.set(ylabel='Installation', xlabel="number of visits", title="Visits by Installation")
        sns.despine(left=True, bottom=True)
        plt.show()

visits_destination(df, otherize_threshold=0.98)

def group_stats(df):
    """For a dataframe print some stats we care about"""
    count = df['Applicant_Age_Visit_Start'].describe()['count']
    mean_age = df['Applicant_Age_Visit_Start'].describe()['mean']
    stdev_age = df['Applicant_Age_Visit_Start'].describe()['std']
    print(f"{count:g} visits. Average age = {mean_age:3.1f}±{stdev_age:3.1f}")
    return

group_stats(df)

def visitor_discipline(df, otherize_threshold=None, json=False):
    """Pass in a dataframe and return a plot of visitors per country.
    If you want the cumulative 95% to apper, and the last 5% to be grouped into other,
    then for example, set otherize_threshol=0.95
    """
    # 'NHM_Specific_Disciplines.SpecificDisciplineName'
    # 'NHM_Disciplines.DisciplineName'
    tmp = Counter(df['NHM_Specific_Disciplines.SpecificDisciplineName']).most_common()
    
    if otherize_threshold:
        top_list_with_other=[]
        other_sum = 0
        percent_limit = int(len(df)*otherize_threshold)
        cumulative = 0
        for k,v in tmp:
            cumulative += v
            if cumulative < percent_limit:
                top_list_with_other.append([k,v])
            else:
                other_sum += v
        top_list_with_other.append(['other', other_sum])
        tmp = top_list_with_other
    if json:
        return dict(tmp)
    else:
        topics = []
        visits = []
        for c,num in tmp:
            topics.append(c)
            visits.append(num)
        discipline_visists = pd.DataFrame(tmp, columns=['disciplines','visits'])
        f, ax = plt.subplots(figsize=(6, 10))
        sns.set_color_codes("pastel")
        sns.barplot(x="visits", y="disciplines", data=discipline_visists,
                    label="Total", color="g")
        ax.set(ylabel='Country', xlabel="number of visits", title="Visits by discipline")
        sns.despine(left=True, bottom=True)
        plt.show()

visitor_discipline(df, otherize_threshold=0.985, json=False)

def visits_per_country(df, otherize_threshold=None):
    """Pass in a dataframe and return a plot of visitors per country"""
    countries = Counter(df['User_NHM.Home_Institution_Country_code']).most_common()
    if otherize_threshold:
        top_list_with_other=[]
        other_sum = 0
        percent_limit = int(len(df)*otherize_threshold)
        cumulative = 0
        for k,v in countries:
            cumulative += v
            if cumulative < percent_limit:
                top_list_with_other.append([k,v])
            else:
                other_sum += v
        top_list_with_other.append(['other', other_sum])
        countries = top_list_with_other
    places = []
    visits = []
    for c,num in countries:
        places.append(c)
        visits.append(num)
    country_visists = pd.DataFrame(countries, columns=['country','visits'])
    f, ax = plt.subplots(figsize=(6, 10))
    sns.set_color_codes("pastel")
    sns.barplot(x="visits", y="country", data=country_visists,
                label="Total", color="b")
    ax.set(ylabel='Country', xlabel="number of visits", title="Visits by country")
    sns.despine(left=True, bottom=True)
    plt.show()

visits_per_country(df, otherize_threshold=0.90)

# In this case I will need to order by the keys, not the values
# Also, I want a displot again here.
Counter(df['ProjectsView.length_of_visit'])

distribution(df, key='ProjectsView.length_of_visit',
             xlabel='length of visit (days)', ylabel='density',
             title="Lenght of visit")

# 'NHM_Specific_Disciplines.SpecificDisciplineName'
# 'NHM_Disciplines.DisciplineName'
filter_var = 'NHM_Specific_Disciplines.SpecificDisciplineName'
for discipline,_ in Counter(df[filter_var]).most_common():
    if isinstance(discipline, str):
        print(f"\nDISCIPLINE: {discipline.capitalize()}")
        mask = df[filter_var] == discipline
        if len(df[mask]) > 100:
            group_stats(df[mask])
            distribution(df[mask], key='Applicant_Age_Visit_Start',
                         ylabel="User age at start of project", xlabel='Age',
                         title=f"Age of users (non-unique) for {discipline}")
            donut_plot(df[mask], column_key='User_NHM.Gender', 
                       fix_keys=["M","F"], colors=['lightskyblue', 'pink'])
            visits_per_country(df[mask], otherize_threshold=0.90)
            visits_destination(df[mask], otherize_threshold=0.98)
            distribution(df[mask], key='ProjectsView.length_of_visit',
                         xlabel='length of visit (days)', ylabel='density',
                         title="Lenght of visit")   
        else:
            print(f"Only {len(df[mask])} entries for discipline. Not sufficent for analysis.")

for country, _ in Counter(df['User_NHM.Home_Institution_Country_code']).most_common():
    print(f"Data grouped for {country}:")
    mask = df['User_NHM.Home_Institution_Country_code'] == country
    if len(df[mask]) > 100:
        group_stats(df[mask])
        donut_plot(df[mask], column_key='User_NHM.Gender', 
                   fix_keys=["M","F"], colors=['lightskyblue', 'pink'])
        distribution(df[mask], key='Applicant_Age_Visit_Start',
                     ylabel="User age at start of project", xlabel='Age',
                     title=f"Age of users (non-unique) from {country}")
        visitor_discipline(df[mask], otherize_threshold=0.98)
        visits_destination(df[mask], otherize_threshold=0.98)
        distribution(df[mask], key='ProjectsView.length_of_visit',
                     xlabel='length of visit (days)', ylabel='density',
                     title="Lenght of visit")   
    else:
        print(f"Only {len(df[mask])} entries for {country}; Not sufficent for meaningful analysis.")
    
    

d={}
for country, _ in Counter(df['User_NHM.Home_Institution_Country_code']).most_common():
    country_d = {}
    if country == country:
        mask = df['User_NHM.Home_Institution_Country_code'] == country
        country_d['sex'] = donut_plot(df[mask], column_key='User_NHM.Gender', json=True)
        country_d['destination'] = visits_destination(df[mask], json=True)
        country_d['discipline']=visitor_discipline(df[mask], json=True)
        d[country] = country_d

#d

#json = json.dumps(d)
#with open("./test_per_country.json","w") as f:
#    f.write(json)



#papers_data = pd.from_xls("/Users/Ben/Work/VIzzuality/SYNTHESIS/Data/synthpubs.xlsx")



