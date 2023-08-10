import requests
r = requests.get('https://api.github.com/users/LucasBerbesson/repos')
data = r.json()

for element in data:
    print(element['language'],":", element['clone_url'])

# Find all alternative softwares
import requests
from IPython.core.display import display, HTML
from bs4 import BeautifulSoup

def find_alternatives(app_name):
    alternatives = []
    print("Searching web for {} alternatives...".format(app_name))
    r = requests.get('http://alternativeto.net/software/'+app_name+'/')
    soup = BeautifulSoup(r.text, "lxml")
    for link in soup.select("article > div.col-xs-2.col-sm-2.col-lg-1 > div.like-box > span"):
        alternatives.append({"name":link.findNext('a')['data-urlname'], "score":link.text})
    
    return alternatives

print(find_alternatives("Python"))

from pprint import pprint
pprint(find_alternatives("ifttt"))

from jinja2 import Environment, FileSystemLoader

env = Environment(loader=FileSystemLoader('templates'))
jinja_template = env.get_template('jinja_demo.html')

def generate_html_page(alternatives,app_name):
    print("Generating a nice HTML Page...")
    return jinja_template.render(alternatives=alternatives, app_name=app_name)

with open("./templates/ifttt.html", "w+") as fh:
        fh.write(generate_html_page(find_alternatives("pycharm"), "ifttt"))



from bottle import route, run, template


@route('/')
def index():
    return 'Go to /alternatives/app_name/'

@route('/alternatives/<app_name>/')
def hello(app_name):
    my_template = generate_html_page(find_alternatives(app_name), app_name)
    return template(my_template)

run(host='localhost', port=8080)

alternatives = find_alternatives("ifttt")

import os
import smtplib 
server = smtplib.SMTP('smtp.gmail.com', 587)
server.starttls()
server.login("lucasberbesson@gmail.com", os.environ['MY_PWD'])
msg = "IFTTT has {} alternatives".format(len(alternatives))
server.sendmail("lucasberbesson@gmail.com","lucas.berbesson@fabdev.fr", msg)
server.quit()

from faker import Faker
f = Faker(locale="fr_FR")
name = f.name()
postcode = f.postcode()
email = f.email()
city = f.city()
color = f.hex_color()
print(name,postcode,email,city,color)

# Script to match all the fabdev.fr emails inside a .txt file
import re
results = []
# Modify the script to match all the email 
for line in open('./data/lorem.txt'):
    results = results + re.findall(r'[\w\.-]+@[\w\.-]+', line)
print(results)

from fuzzywuzzy import fuzz
print("Simple ratio : ",fuzz.ratio("this is a test", "this is a test!"))
print("Partial ratio : ",fuzz.partial_ratio("this is a test", "this is a test!"))
print("Token Ration : ",fuzz.token_sort_ratio("fuzzy wuzzy was a bear", "wuzzy fuzzy was a bear"))

from fuzzywuzzy import fuzz
from fuzzywuzzy import process
state_to_code = {"VERMONT": "VT", "GEORGIA": "GA", "IOWA": "IA", "Armed Forces Pacific": "AP", "GUAM": "GU",
                 "KANSAS": "KS", "FLORIDA": "FL", "AMERICAN SAMOA": "AS", "NORTH CAROLINA": "NC", "HAWAII": "HI",
                 "NEW YORK": "NY", "CALIFORNIA": "CA", "ALABAMA": "AL", "IDAHO": "ID", "FEDERATED STATES OF MICRONESIA": "FM",
                 "Armed Forces Americas": "AA", "DELAWARE": "DE", "ALASKA": "AK", "ILLINOIS": "IL",
                 "Armed Forces Africa": "AE", "SOUTH DAKOTA": "SD", "CONNECTICUT": "CT", "MONTANA": "MT", "MASSACHUSETTS": "MA",
                 "PUERTO RICO": "PR", "Armed Forces Canada": "AE", "NEW HAMPSHIRE": "NH", "MARYLAND": "MD", "NEW MEXICO": "NM",
                 "MISSISSIPPI": "MS", "TENNESSEE": "TN", "PALAU": "PW", "COLORADO": "CO", "Armed Forces Middle East": "AE",
                 "NEW JERSEY": "NJ", "UTAH": "UT", "MICHIGAN": "MI", "WEST VIRGINIA": "WV", "WASHINGTON": "WA",
                 "MINNESOTA": "MN", "OREGON": "OR", "VIRGINIA": "VA", "VIRGIN ISLANDS": "VI", "MARSHALL ISLANDS": "MH",
                 "WYOMING": "WY", "OHIO": "OH", "SOUTH CAROLINA": "SC", "INDIANA": "IN", "NEVADA": "NV", "LOUISIANA": "LA",
                 "NORTHERN MARIANA ISLANDS": "MP", "NEBRASKA": "NE", "ARIZONA": "AZ", "WISCONSIN": "WI", "NORTH DAKOTA": "ND",
                 "Armed Forces Europe": "AE", "PENNSYLVANIA": "PA", "OKLAHOMA": "OK", "KENTUCKY": "KY", "RHODE ISLAND": "RI",
                 "DISTRICT OF COLUMBIA": "DC", "ARKANSAS": "AR", "MISSOURI": "MO", "TEXAS": "TX", "MAINE": "ME"}

process.extractOne("Minesotta",choices=state_to_code.keys())

process.extractOne("AlaBAMMazzz",choices=state_to_code.keys(),score_cutoff=80)

import numpy as np

x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])
print(np.dot(x, y))
print(x.T)
print(x.sum())
np.arange(10,51)

np.eye(3)

np.linspace(0,1,20).reshape(4,5)

import pandas as pd

df = pd.read_excel('./data/data.xlsx')

df.head()

# Get list of values in a column
df['PROJET'].unique()

# Filter the dataframe with a constraint on a column
df[df['PROJET']=="Convergence"]

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')

plt.figure(figsize=(12,15))
sns.countplot(y='PROJET',data=df);

plt.figure(figsize=(12,5))
sns.countplot(x='PROJET',hue='STATUT',data=df[df['PROJET'].isin(['Convergence', 'Apogée','Antares','EOD','imaGrid'])]);



