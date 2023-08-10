get_ipython().magic('matplotlib inline')
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
import time
pd.set_option("display.width", 500)
pd.set_option("display.max_columns", 100)
pd.set_option("display.notebook_repr_html", True)
import seaborn as sns
sns.set_style("darkgrid")
sns.set_context("poster")

# read csv of syllabi to dataframe
justia_df = pd.read_csv("final_justia_data_merge.csv")
justia_df.head()

# ---------- randomly sample 20% of the justia data ----------

from random import sample
print "total number of cases: ", len(justia_df)

# create a unique case_id for each case
justia_df["case_id"] = justia_df.index

# randomly sample to get 20% of the cases 
sample_df = justia_df.loc[sample(justia_df.index, len(justia_df)/5)]
sample_df = sample_df.reset_index(drop=True)

# drop year column (will be duplicate)
sample_df = sample_df.drop("year",1)

# get length of syllabi to determine which syllabi are undecided cases (ie, very short syllabi)
hist_textlength,textlength = [],[]
for index, row in sample_df.iterrows():
    l = len(str(row["text"]))
    textlength.append((l,row["case_id"]))
    hist_textlength.append(l)

# plot histogram of word length in chars
plt.figure(figsize=(22,10))
plt.grid(zorder=3)
plt.gca().xaxis.grid(False)
plt.hist(hist_textlength,bins=np.arange(0, 25000, 800),color="#77dbc9",linewidth=0,zorder=0)
plt.title("Distribution of Syllabi by Length (in Number of Characters)")
plt.yticks(np.arange(0,600,50))
plt.ylabel("Number of Cases")
plt.xlabel("Word Length (Characters)")

# print percentiles
p5 = np.percentile(hist_textlength, 5) 
p25 = np.percentile(hist_textlength, 25) 
p50 = np.percentile(hist_textlength, 50) 
p75 = np.percentile(hist_textlength, 75) 
p95 = np.percentile(hist_textlength, 95) 
print "5th percentile: ", p5
print "25th percentile: ", p25
print "50th percentile: ", p50
print "75th percentile: ", p75
print "95th percentile: ", p95

# function to find the text length closest to a given length 
closest = lambda num: min(hist_textlength, key=lambda x:abs(x-num))
textlength_dict = dict(textlength)

# we find the case id corresponding to the closest number, and print the corresponding text
test_lengths = [200,300,400,500,600,700,800]
for length in test_lengths:
    print "%d characters: " % length + "\n----------"
    print sample_df[sample_df["case_id"]==textlength_dict[closest(length)]].iloc[0]["text"] + "\n"

# exclude cases with fewer than 400 characters
for case in textlength:
    if case[0]<400: 
        sample_df = sample_df[sample_df["case_id"]!=case[1]]      

sample_df = sample_df.reset_index(drop=True)

# convert csv to dataframe, keep useful columns
scdb_df = pd.read_csv("SCDB_2015_01_caseCentered_Citation.csv")
scdb_df = scdb_df[['caseId','docketId','usCite','docket','caseName','dateArgument','caseOriginState','jurisdiction',                   'issueArea','decisionDirection','decisionType','lawType','majOpinWriter','majVotes','minVotes']]

scdb_df.tail()

# ---------- create a list of U.S. cites and volumes to map to SCDB data ----------

from bs4 import BeautifulSoup
import requests as rq
import re

# wikipedia lists all scotus cases by volume; scrape docket number to match u.s. cite format
url_stem = "https://en.wikipedia.org/wiki/List_of_United_States_Supreme_Court_cases,_volume_"

# save U.S. cites and docket numbers
cites,cite_nums = [],[]

# for wikipedia pages before vol. 565, scrape unstructured list
for vol in range(558,565):
    soup = BeautifulSoup(rq.get(url_stem + str(vol)).text, "lxml")
    caselist = soup.find("div", attrs={"id":"mw-content-text"}).ul
    for case in caselist.findAll("li"):
        try:
            cite_num = case.text.split("No. ")[1][:-1]
        except IndexError:
            cite_num = case.text.split(", ")[1][:-1]
        cite_nums.append(cite_num)
        cites.append(str(vol) + " U.S. " + cite_num)

# for wikipedia pages vol. 565 and after, scrape table
for vol in range(565,576):
    soup = BeautifulSoup(rq.get(url_stem + str(vol)).text, "lxml")
    caselist = soup.find("table", attrs={"class":"wikitable plainlinks"})
    for cite in caselist.findAll("td", attrs={"style":"white-space:nowrap;"}):
        cite_num = cite.text
        cite_nums.append(cite_num)
        cites.append(str(vol) + " U.S. " + cite_num)

# create dataframe to merge U.S. cites 
cite_dict = {}
cite_dict["us_cite"] = cites
cite_dict["docket_num"] = cite_nums
citedf = pd.DataFrame(cite_dict)

# add a year column to SCDB data in order to subset 2010+ cases
scdb_df["year"] = [int(scdb_df["caseId"][x].split("-")[0]) for x in range(0,len(scdb_df))]
scdb_subset = scdb_df[scdb_df["year"] >= 2010]
scdb_merged = pd.merge(scdb_subset, citedf, left_on='docket', right_on='docket_num')

# merge the scraped mapped list with the 2010+ SCDB data
scdb_merged = scdb_merged.drop(["docket_num","usCite"],axis=1)
scdb_merged.rename(columns={"us_cite":"usCite"}, inplace=True)
scdb_merged = scdb_merged[['caseId','docketId','usCite','docket','caseName','dateArgument','caseOriginState','jurisdiction',                   'issueArea','decisionDirection','decisionType','lawType','majOpinWriter','majVotes','minVotes','year']]

# concatenate the updated 2010+ and pre-2010 SCDB data
final_scdb = pd.concat([scdb_df,scdb_merged])
final_scdb = final_scdb.reset_index()
final_scdb.tail()

# now we can successfully merge the Justia and SCDB data on U.S. cite
full_df = pd.merge(sample_df, final_scdb, left_on='us_cite', right_on='usCite')
full_df = full_df.drop(["index"],1)

# check that 2010+ cases are included
full_df.sort("year").tail()

# simplify data 
full_df = full_df.drop("caseName",1)

# save sample dataframe for various models
full_df.to_csv("sample_cases.csv", sep=',', encoding='utf-8',index=False)

justia_df = justia_df.drop("year",1)

# exclude cases with fewer than 400 characters
for case in textlength:
    if case[0]<400: 
        justia_df = justia_df[justia_df["case_id"]!=case[1]]      

justia_df = justia_df.reset_index(drop=True)

# merge the full Justia data set and SCDB data on U.S. cite
full_justia_df = pd.merge(justia_df, final_scdb, left_on='us_cite', right_on='usCite')
full_justia_df = full_justia_df.drop(["index"],1)

# simplify data 
full_justia_df = full_justia_df.drop("caseName",1)

# save sample dataframe for various models
full_justia_df.to_csv("all_cases.csv", sep=',', encoding='utf-8',index=False)

# plot distribution of all cases by year
plt.figure(figsize=(22,10))
plt.grid(zorder=3)
plt.gca().xaxis.grid(False)
plt.hist(scdb_df.year, bins=np.arange(min(scdb_df.year),max(scdb_df.year),1),color="#53d8ce",linewidth=0,zorder=0)
plt.title("Distribution of Syllabi by Year")
plt.yticks(np.arange(0,200,10))
plt.ylabel("Number of Cases")
plt.xlabel("Years")
plt.xlim((min(scdb_df.year),max(scdb_df.year)-1))

# plot court eras for context
plt.axvline(1953,color="#f4593d",linestyle="dashed")
plt.text(1953.5,80,"Warren Court",rotation=90,size=17)
plt.axvline(1969,color="#f4593d",linestyle="dashed")
plt.text(1969.5,80,"Burger Court",rotation=90,size=17)
plt.axvline(1986,color="#f4593d",linestyle="dashed")
plt.text(1986.5,80,"Rehnquist Court",rotation=90,size=17)
plt.axvline(2005,color="#f4593d",linestyle="dashed")
plt.text(2005.5,80,"Roberts Court",rotation=90,size=17)

plt.show()

# turn issue areas into dummy column
issue_areas = ["criminal procedure","civil rights","first amendment","due process","privacy","attorneys",
              "unions","economic activity","judicial power","federalism","interstate  amendment",
              "federal taxation","miscellaneous","private action"]

for issue, num in zip(issue_areas,range(1,15)):
    scdb_df[issue] = scdb_df.issueArea.apply(lambda x: 1 if x == num else 0)

# plot distribution of all cases by issue type
import collections

issue_dict = {}
for issue in issue_areas:
    issue_dict[issue] = sum(scdb_df[issue])

sorted_dict = collections.OrderedDict()
sorted_vals = sorted(issue_dict.values(),reverse=True)
sorted_keys = sorted(issue_dict, key=issue_dict.get,reverse=True)
for key, val in zip(sorted_keys,sorted_vals):
    sorted_dict[key] = val

plt.figure(figsize=(20,12))
plt.grid(zorder=3)
plt.barh(range(len(sorted_dict)),sorted_dict.values(),align='center',color=sns.color_palette("Set2", 14),linewidth=0,zorder=0)
plt.gca().yaxis.grid(False)
plt.gca().invert_yaxis()
plt.yticks(range(len(sorted_dict)),sorted_dict.keys())
plt.xticks(np.arange(0,2000,200))
plt.title("Distribution of Syllabi by Issue Area")
plt.xlabel("Number of Cases")
plt.ylabel("Issue Areas")
plt.show()

# turn decision directions into dummy column (conservative, liberal, neutral)
decision_areas = ["conservative","liberal","neutral"]

for decision, num in zip(decision_areas,range(1,4)):
    scdb_df[decision] = scdb_df.decisionDirection.apply(lambda x: 1 if x == num else 0)

# turn decision type into dummy column
for num in range(1,9):
    scdb_df["dec_" + str(num)] = scdb_df.decisionType.apply(lambda x: 1 if x == num else 0)

# plot distribution of all cases by issue type
types = ["opinion of the court (orally argued)", "per curiam (no oral argument)", "decrees", "equally divided vote",         "per curiam (orally argued)", "judgment of the Court (orally argued)", "seriatim"]
type_dict = {}
for dtype,label in zip(range(1,9),types):
    type_dict[label] = sum(scdb_df["dec_" + str(dtype)])

sorted_type_dict = collections.OrderedDict()
sorted_type_vals = sorted(type_dict.values(),reverse=True)
sorted_type_keys = sorted(type_dict, key=type_dict.get,reverse=True)
for key, val in zip(sorted_type_keys,sorted_type_vals):
    sorted_type_dict[key] = val

plt.figure(figsize=(20,10))
plt.grid(zorder=3)
plt.barh(range(len(sorted_type_dict)),sorted_type_dict.values(),align='center',color=sns.color_palette("Set2", 14),linewidth=0,zorder=0)
plt.gca().yaxis.grid(False)
plt.gca().invert_yaxis()
plt.yticks(range(len(sorted_type_dict)),sorted_type_dict.keys())
plt.title("Distribution of Syllabi by Decision Type")
plt.xlabel("Number of Cases")
plt.ylabel("Decision Types")
plt.show()

# plot distribution of all cases by year
plt.figure(figsize=(22,10))
plt.grid(zorder=3)
plt.gca().xaxis.grid(False)
plt.hist(scdb_df.year, bins=np.arange(min(scdb_df.year),max(scdb_df.year),1),color="#53d8ce",linewidth=0,alpha=0.4,zorder=0,label="Full Data Set")
plt.hist(full_df.year, bins=np.arange(min(full_df.year),max(full_df.year),1),color="#53d8ce",linewidth=0,zorder=0,label="Random Sample")
plt.title("Distribution of Syllabi by Year (Full Data Set vs. Random Sample)")
plt.yticks(np.arange(0,200,10))
plt.ylabel("Number of Cases")
plt.xlabel("Years")
plt.xlim((min(scdb_df.year),max(scdb_df.year)-1))
plt.legend()

# plot court eras for context
plt.axvline(1953,color="#f4593d",linestyle="dashed")
plt.text(1953.5,80,"Warren Court",rotation=90,size=17)
plt.axvline(1969,color="#f4593d",linestyle="dashed")
plt.text(1969.5,80,"Burger Court",rotation=90,size=17)
plt.axvline(1986,color="#f4593d",linestyle="dashed")
plt.text(1986.5,80,"Rehnquist Court",rotation=90,size=17)
plt.axvline(2005,color="#f4593d",linestyle="dashed")
plt.text(2005.5,80,"Roberts Court",rotation=90,size=17)

plt.show()

# turn issue areas into dummy column
for num in range(1,9):
    full_df["dec_" + str(num)] = full_df.decisionType.apply(lambda x: 1 if x == num else 0)
    
# plot distribution of all cases by issue type
samp_issue_dict = {}
for issue in issue_areas:
    samp_issue_dict[issue] = sum(full_df[issue])

samp_sorted_dict = collections.OrderedDict()
samp_sorted_vals = sorted(samp_issue_dict.values(),reverse=True)
samp_sorted_keys = sorted(samp_issue_dict, key=issue_dict.get,reverse=True)
for key, val in zip(samp_sorted_keys,samp_sorted_vals):
    samp_sorted_dict[key] = val

plt.figure(figsize=(20,12))
plt.grid(zorder=3)
plt.barh(range(len(sorted_dict)),sorted_dict.values(),align='center',color=sns.color_palette("Set2", 14),alpha=0.4,linewidth=0,zorder=0,label="Full Data Set")
plt.barh(range(len(samp_sorted_dict)),samp_sorted_dict.values(),align='center',color=sns.color_palette("Set2", 14),linewidth=0,zorder=0,label="Random Sample")
plt.gca().yaxis.grid(False)
plt.gca().invert_yaxis()
plt.yticks(range(len(sorted_dict)),sorted_dict.keys())
plt.title("Distribution of Syllabi by Issue Area (Random Sample)")
plt.xlabel("Number of Cases")
plt.ylabel("Issue Areas")
plt.legend(loc="center right")
plt.xticks(np.arange(0,2000,200))
plt.show()

# turn decision types into dummy column
for num in range(1,9):
    full_df["dec_" + str(num)] = full_df.decisionType.apply(lambda x: 1 if x == num else 0)

# plot distribution of all cases by issue type
samp_type_dict = {}
for dtype,label in zip(range(1,9),types):
    samp_type_dict[label] = sum(full_df["dec_" + str(dtype)])

samp_sorted_type_dict = collections.OrderedDict()
samp_sorted_type_vals = sorted(samp_type_dict.values(),reverse=True)
samp_sorted_type_keys = sorted(samp_type_dict, key=samp_type_dict.get,reverse=True)
for key, val in zip(samp_sorted_type_keys,samp_sorted_type_vals):
    samp_sorted_type_dict[key] = val

plt.figure(figsize=(20,10))
plt.grid(zorder=3)
plt.barh(range(len(sorted_type_dict)),sorted_type_dict.values(),align='center',color=sns.color_palette("Set2", 14),alpha=0.4,linewidth=0,zorder=0,label="Full Data Set")
plt.barh(range(len(samp_sorted_type_dict)),samp_sorted_type_dict.values(),align='center',color=sns.color_palette("Set2", 14),linewidth=0,zorder=0,label="Random Sample")
plt.gca().yaxis.grid(False)
plt.gca().invert_yaxis()
plt.yticks(range(len(sorted_type_dict)),sorted_type_dict.keys())
plt.title("Distribution of Syllabi by Decision Type (Random Sample)")
plt.xlabel("Number of Cases")
plt.ylabel("Decision Types")
plt.legend(loc="center right")
plt.show()

# plot distribution of partisanship by issue area 

# create subplot of 4 rows x 4 cols
fig, ax = plt.subplots(4, 4, figsize=(24, 20), tight_layout=True)

# index of issue area
x = 1

# iterate through rows, cols of subplot grid, populate with kde
for r in range(0,4):
    for c in range(0,4):
        ax[r][c].set_title(issue_areas[x-1])
        
        decision_dict = {}
        for decision in decision_areas:
            decision_dict[decision] = sum(scdb_df[scdb_df["issueArea"] == x][decision])

        sorted_dec_dict = collections.OrderedDict()
        sorted_dec_vals = sorted(decision_dict.values(),reverse=True)
        sorted_dec_keys = sorted(decision_dict, key=decision_dict.get,reverse=True)
        for key, val in zip(sorted_dec_keys,sorted_dec_vals):
            sorted_dec_dict[key] = val
            
        ax[r][c].grid(zorder=3)
        ax[r][c].barh(range(len(sorted_dec_dict)),sorted_dec_dict.values(),align='center',color=["#2fa6ff","#ff592f","#fbe485"],height=0.5,linewidth=0,zorder=0)
        ax[r][c].set_yticks([0,1,2])
        ax[r][c].set_xticks(np.arange(0,max(sorted_dec_dict.values()),max(sorted_dec_dict.values())/2))
        ax[r][c].set_xlabel("Number of Cases")
        ax[r][c].set_ylabel("Decision Direction")

        x = x + 1
        
        if x == 15:
            break

plt.show()



