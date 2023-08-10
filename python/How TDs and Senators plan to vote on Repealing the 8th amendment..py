import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
import folium
import json
from pprint import pprint

data = pd.read_csv("Irish_Times.csv")
data.head()

Constituency = data["Constituency"]
Vote_Yes = []
Vote_No =[]
Vote_Und = []
Voter_Group = []

data_newer = [Constituency,Vote_Yes,Vote_No]

for i in data["Vote"]:
    if i == "Yes":
        Vote_Yes.append(1)
        Vote_No.append(0)
        Vote_Und.append(0)
    elif i == "No":
        Vote_Yes.append(0)
        Vote_No.append(1)
        Vote_Und.append(0)
    elif i =="Undeclared":
        Vote_Yes.append(0)
        Vote_No.append(0)
        Vote_Und.append(1)

for i in data["Voter type"]:
    if i == "TD":
        Voter_Group.append("Dáil")
    else: 
        Voter_Group.append("Seanad")
        
data_new = pd.DataFrame(Constituency)   
data_new["Yes"] = Vote_Yes
data_new["No"] = Vote_No
data_new["Undecided"] = Vote_Und
data_new["Party"] = data["Party"]
data_new["Voter type"] = data["Voter type"]
data_new["Voter_Group"] = Voter_Group

print ("Total Dail and Seanad votes (As of 20/01/2018)")
print("Yes: ", sum(Vote_Yes) )
print("No:", sum(Vote_No) )
print("Undeclared:", sum(Vote_Und))

data_new.groupby( [ "Voter_Group"] ).sum()

data_new.groupby( [ "Party"] ).sum()

data_new.groupby( [ "Constituency","Voter_Group"] ).sum()

