import pandas as pd
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2
get_ipython().magic('matplotlib inline')

# Plain Seaborn figures with matplotlib color codes mapped to the default seaborn palette 
sns.set(style="white", color_codes=True)

gender_rate = {}
gender_factor = {}
gender_number = {}
gender_rate["Male"] = 278.4e-5
gender_rate["Female"] = 627.2e-5
gender_number["Male"] = 155651602
gender_number["Female"] = 160477237
rate_average = (gender_rate["Male"]*gender_number["Male"]+gender_rate["Female"]*gender_number["Male"])/(gender_number["Male"]+gender_number["Female"])
gender_factor["Male"] = gender_rate["Male"]/rate_average
gender_factor["Female"] = gender_rate["Female"]/rate_average
gender_factor["Female"], gender_factor["Male"]
race_rate = {}
race_factor = {}
race_number = {}
race_number["Native"] = 1942876.0
race_number["Asian"] = 12721721.0
race_number["Black"] = 29489649.0
race_number["Hispanic"] = 46407173.0
race_number["Multiple"] = 5145135.0
race_number["Pacific"] = 473703.0
race_number["White"] = 161443167.0
race_rate["Native"] = 689.1e-5
race_rate["Asian"] = 115.8e-5
race_rate["Black"] = 1152.6e-5
race_rate["Hispanic"] = 376.2e-5
race_rate["Multiple"] = 116.1e-5
race_rate["Pacific"] = 641.5e-5
race_rate["White"] = 187.0e-5
US_number = race_number["Native"] + race_number["Asian"] + race_number["Black"] + race_number["Hispanic"] + race_number["Multiple"] + race_number["Pacific"] + race_number["White"]
rate_average = (race_rate["Native"]*race_number["Native"]+race_rate["Asian"]*race_number["Asian"]+race_rate["Black"]*race_number["Black"]+race_rate["Hispanic"]*race_number["Hispanic"]+race_rate["Multiple"]*race_number["Multiple"]+race_rate["Pacific"]*race_number["Multiple"]+race_rate["White"]*race_number["White"])/US_number  
race_factor["Native"] = race_rate["Native"]/rate_average
race_factor["Asian"] = race_rate["Asian"]/rate_average
race_factor["Black"] = race_rate["Black"]/rate_average
race_factor["Hispanic"] = race_rate["Hispanic"]/rate_average
race_factor["Multiple"] = race_rate["Multiple"]/rate_average
race_factor["Pacific"] = race_rate["Pacific"]/rate_average
race_factor["White"] = race_rate["White"]/rate_average

age_rate = {}
age_factor = {}
age_number = {}
age_number["0-14"] = 61089123.0
age_number["15-19"] = 21158964.0
age_number["20-24"] = 22795438.0
age_number["25-29"] = 21580198.0
age_number["30-34"] = 21264389.0
age_number["35-39"] = 19603770.0
age_number["40-44"] = 20848920.0
age_number["45-54"] = 43767532.0
age_number["55-64"] = 39316431.0
age_number["65+"] = 44704074.0

age_rate["0-14"] = 20.0e-5
age_rate["15-19"] = 1804.0e-5
age_rate["20-24"] = 2484.6e-5
age_rate["25-29"] = 1176.2e-5
age_rate["30-34"] = 532.4e-5
age_rate["35-39"] = 268.0e-5
age_rate["40-44"] = 131.5e-5
age_rate["45-54"] = 56.6e-5
age_rate["55-64"] = 16.6e-5
age_rate["65+"] = 3.2e-5

US_age_number = age_number["0-14"] + age_number["15-19"] + age_number["20-24"] + age_number["25-29"] + age_number["30-34"] + age_number["35-39"] + age_number["40-44"] + age_number["45-54"] + age_number["55-64"] + age_number["65+"]
rate_average = (age_rate["0-14"]*age_number["0-14"]+age_rate["15-19"]*age_number["15-19"]+age_rate["20-24"]*age_number["20-24"]+age_rate["25-29"]*age_number["25-29"]+age_rate["30-34"]*age_number["30-34"]+age_rate["35-39"]*age_number["35-39"]+age_rate["40-44"]*age_number["40-44"]+age_rate["45-54"]*age_number["45-54"]+age_rate["55-64"]*age_number["55-64"]+age_rate["65+"]*age_number["65+"])/US_age_number  
age_factor["0-14"] = age_rate["0-14"]/rate_average
age_factor["15-19"] = age_rate["15-19"]/rate_average
age_factor["20-24"] = age_rate["20-24"]/rate_average
age_factor["25-29"] = age_rate["25-29"]/rate_average
age_factor["30-34"] = age_rate["30-34"]/rate_average
age_factor["35-39"] = age_rate["35-39"]/rate_average
age_factor["40-44"] = age_rate["40-44"]/rate_average
age_factor["45-54"] = age_rate["45-54"]/rate_average
age_factor["55-64"] = age_rate["55-64"]/rate_average
age_factor["65+"] = age_rate["65+"]/rate_average

race_factor["Native"], race_factor["Asian"], race_factor["Black"], race_factor["Hispanic"], race_factor["Multiple"], race_factor["Pacific"], race_factor["White"]
age_factor["0-14"], age_factor["15-19"], age_factor["20-24"], age_factor["25-29"], age_factor["30-34"], age_factor["35-39"], age_factor["40-44"], age_factor["45-54"], age_factor["55-64"], age_factor["65+"]

model = pickle.load(open('../data/randomforest_params.pickle', "rb" ))

Ymean = pickle.load(open('../data/Ymean.pickle', "rb"))

Ystd = pickle.load(open('../data/Ystd.pickle', "rb"))

dbname = 'census_zipcode_db'
username = 'akuepper'
pswd = 'FLP@nd'

engine = create_engine('postgresql://%s:%s@localhost/%s'%(username,pswd,dbname))
print(engine.url)

# connect:
con = None
con = psycopg2.connect(database = dbname, user = username, host='localhost', password=pswd)

# query:
sql_query = """
SELECT * FROM zip_census_db WHERE geoid2='602';
"""

data_from_sql = pd.read_sql_query(sql_query,con)

data_from_sql.Population[0]

sql_query = """
SELECT * FROM zip_census_db;
"""
data_from_sql = pd.read_sql_query(sql_query,con)
data_from_sql.head()

def calculate_rate(sql_query, con):
    target = pd.read_sql_query(sql_query,con)
    target_params = target.values[0]
    chlamydia_rate = model.predict(target_params[1:])*Ystd+Ymean
    return chlamydia_rate

Race = "White"
Gender = "Male"
Age = "15-19"
Zipcode = "02139"

sql_query = "SELECT * FROM zip_census_db WHERE geoid2=%i;"%(int(Zipcode))

calculate_rate(sql_query, con)

# query:
sql_query = "SELECT * FROM zip_census_unnormalized_db WHERE geoid2=%i;"%(int(Zipcode))

target_unnormalized = pd.read_sql_query(sql_query,con)

TOTALNR = target_unnormalized["Population"]

if Gender == "Male":
    gender_table = "hd02s026"
else:
    gender_table = "hd02s051"

GENDERNR = TOTALNR*target_unnormalized[gender_table]/100.0

if Race == "White":
    race_table = "hd02s078"
elif Race == "Black":
    race_table = "hd02s079"
elif Race == "Native":
    race_table = "hd02s080"
elif Race == "Asian":
    race_table = "hd02s081"
elif Race == "Pacific":
    race_table = "hd02s089"
elif Race == "Multiple":
    race_table = "hd02s095"
elif Race == "Hispanic":
    race_table = "hd02s107"

RACENR = TOTALNR*target_unnormalized[race_table]/100.0

if Age == "0-14":
    age_table = "hd02s002"
elif Age == "15-19":
    age_table = "hd02s005"
elif Age == "20-24":
    age_table = "hd02s006"
elif Age == "25-29":
    age_table = "hd02s007"
elif Age == "30-34":
    age_table = "hd02s008"
elif Age == "35-39":
    age_table = "hd02s009"
elif Age == "40-44":
    age_table = "hd02s010"
elif Age == "45-54":
    age_table = "hd02s011"
elif Age == "55-64":
    age_table = "hd02s013"
elif Age == "65+":
    age_table = "hd02s015"

AGENR = TOTALNR*target_unnormalized[age_table]/100.0

sql_query = "SELECT * FROM zip_census_db WHERE geoid2=%i;"%(int(Zipcode))

zipcoderate = calculate_rate(sql_query, con)*100
genderrate = gender_rate[Gender]*100
agerate = age_rate[Age]*100
racerate = race_rate[Race]*100

the_result = (zipcoderate/TOTALNR.values + genderrate/GENDERNR.values + racerate/RACENR.values + agerate/AGENR.values)/(1.0/TOTALNR.values+1.0/GENDERNR.values+1.0/RACENR.values+1.0/AGENR.values)

d = np.array([the_result[0], genderrate, agerate, racerate, zipcoderate[0]])
d_label = np.array(["You", "Your gender", "Your age group", "Your race / ethnicity", "Your location"])
d_label

sns.set(style="white", context="talk")

fig, ax = plt.subplots(1, 1, figsize=(10, 6), sharex=True)
sns.barplot(d_label, d, palette="RdBu_r", ax=ax)
ax.set_ylabel("Risk", fontsize=20)
plt.title(r'Chlamydia', fontsize=20)
ax.plot([-1, len(d)], [0,0], "k-", linewidth=1.0)
sns.despine(bottom=True)
plt.setp(fig.axes, yticks=[])
plt.tight_layout(h_pad=3)



