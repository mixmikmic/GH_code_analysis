import pandas as pd
import string as st

import matplotlib
import numpy as np
from datascience import *

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

# Load data CSV -> datascience Table

data_file = '../data/180213_cogsci_journal_unprocessed.csv'
df_pd = pd.read_csv(data_file)

#df = Table.read_table(data_file)
df_ad = df_pd['authors'][0:]
df_pd

df_pf.rename()
tbl = df_pd[["year", 0, "article_name", "article_type"]]
#index_paperName = Table().with_columns(["Year",df.column(2),
#                                        "Paper Index", df.column(0),
#                                        "Article Name", df.column(1),
#                                        "Old Type", df.column(9)])

# create list of lists where the number in each sublist is the article index and the length of the 
# sublist is the number of authors for that article
# - i+1 to counter-act data indexing
df_namesplit = []
df_numauth = []
for i in range(1, len(df_ad)):
        
    temp = df_ad[i].split('/n')
    df_namesplit.append(temp)
    df_numauth.append([i+1]*len(temp))

#print(df_numauth)

# Makes list of lists into single list
finList = []
for i in range(len(df_numauth)):
    for j in range(len(df_numauth[i])):
        finList.append(df_numauth[i][j])
        
#print(finList)

finList.to_csv("Num_Authors_Per_Article")

# This function takes a string (author of article ) and strips "Corresponding Author" -> returns stripped string
def clearCorresponding(author):
    result = []
    if ',Corresponding author' in author:

        endIndex = author.find(',')
        result.append(author[:endIndex])
        return result
    
    elif 'Corresponding author' in author:
        endIndex = author.find('Corresponding author')
        
        result.append(author[:endIndex])
        return result
   
    else:
        return author

# Main algorithm to construct table of Articles and Authors

results = Table(make_array("Article Id", "Author", "Department/University", "Corresponding Author"))
# For each sublist in the list of lists df_namesplit
articleCounter = 0 # Used to iterate through list of article id's
for i in range(len(df_namesplit)):
    
    articlesAuthors = df_namesplit[i]
    
    for j in range(len(articlesAuthors)):
        corr_auth = 0
        element = df_namesplit[i][j].split('             ')
        if len(element) > 1: # If the article has an author and the corresponding department/ university
    
            author = clearCorresponding(element[0])
            dept = element[1]
            
            if("Corresponding author" in df_namesplit[i][j]):
                corr_auth = 1
            results = results.with_row([finList[articleCounter],author, dept, corr_auth])
            articleCounter += 1
            
        else:            
            author = clearCorresponding(element[0])
            dept = "NaN"
            
            #newRow = make_array(corr_auth,author, dept)
            results = results.with_row([finList[articleCounter],author, dept, corr_auth])
            articleCounter += 1
            

# Join article name with article id number
results = results.join("Article Id", index_paperName, "Paper Index")
# not sure why we need to drop column 5 twice
#results.show()

Extended_Article = ["Extended Articles", "Extended Article", "Extended article", "Extended articles"]
Original_Article = ["Original Article", "Original Articles"]
Regular_Article = ["Regular Articles", "Regular articles", "Regular Article", "Regular article"]
Letter_to_Editor = ["Letters to the Editor","Letter to the Editor"]
Brief_Report = ["Brief Reports", "Brief Report", "Brief reports", "Brief report"]
Announcement = ["Announcement"]
Special = ["2011 Rumelhart Prize Special Issue Honoring Judea Pearl",
           "SPECIAL ISSUE: 2009 Rumelhart Prize Special Issue Honoring Susan Carey",
           "2010 Rumelhart Prize Special Issue Honoring James L. McClelland",
           "SPECIAL ISSUE: 2007 Rumelhart Prize Special Issue Honoring Jeffrey L. Elman Language as a Dynamical System"]
Comment = ["Comment","Short Communication"]
Commentary = ["Commentary"]

def consolidate_Type (oldType):
    # put the definitions inside the function, but ideally in a separate data file
#     Extended_Article = ["Extended Articles", "Extended Article", "Extended article", "Extended articles"]
#     Original_Article = ["Original Article", "Original Articles"]
#     Regular_Article = ["Regular Articles", "Regular articles", "Regular Article", "Regular article"]
#     Letter_to_Editor = ["Letters to the Editor","Letter to the Editor"]
#     Brief_Report = ["Brief Reports", "Brief Report", "Brief reports", "Brief report"]
#     Announcement = ["Announcement"]
#     Special = ["2011 Rumelhart Prize Special Issue Honoring Judea Pearl",
#                "SPECIAL ISSUE: 2009 Rumelhart Prize Special Issue Honoring Susan Carey",
#                "2010 Rumelhart Prize Special Issue Honoring James L. McClelland",
#                "SPECIAL ISSUE: 2007 Rumelhart Prize Special Issue Honoring Jeffrey L. Elman Language as a Dynamical System"]
#     Comment = ["Comment","Short Communication"]
#     Commentary = ["Commentary"]
    if oldType in Extended_Article:
        return "Extended Article"
    if oldType in Original_Article:
        return "Original Article"
    if oldType in Regular_Article:
        return "Regular Article"
    if oldType in Brief_Report:
        return "Brief Report"
    if oldType in Letter_to_Editor:
        return "Letter to Editor"
    if oldType in Announcement:
        return "Announcement"
    if oldType in Special:
        return "Special"
    if oldType in Comment:
        return "Comment"
    if oldType in Commentary:
        return "Commentary"
    return "null"

results = results.with_column("Article Type", results.apply(consolidate_Type, "Old Type"))
results = results.drop("Old Type").drop("Year_2").drop("Article Name_2")

results = results.drop("Year_2").drop("Article Name_2").drop("New Type")
results

results2 = Table().with_columns(["Article Id",[],
                                 "Author", [], 
                                 "Department/University", [], 
                                 "Corresponding Author",[],
                                 "Year",[],
                                 "Article Name", [],
                                 "New Type", [] ])

for i in range(results.num_rows):
    if results.column(5)[i] != "null":
        results2 = results2.with_row(results.row(i))  
#results2.show()

results2.to_csv("cleaned_data")

# issue Computational Neuroscience
compSci = ["artificial intelligence", "computer science", "informatics", "computer", "cybernetics", "computing", "eecs", "technology"]

def consolodateAffiliation(aff):
    temp = aff.lower()
    
    if "cognitive" in temp or "cognition" in temp:
        return "Cognitive Science"
    if "psychology" in temp or "psychological" in temp:
        return "Psychology"
    if "philosophy" in temp:
        return "Philosophy"
    if "linguistics" in temp:
        return "Linguistics"
    if "neuro" in temp or "brain" in temp:
        return "Neuroscience"
    for i in compSci:
        if i in temp:
            return "Artificial Intelligence"
    if "anthropology" in temp:
        return "Anthropology"
    if "social science" in temp:
        return "Social Science"
    return "Other"

results2 = results2.with_column("Affiliation",results2.apply(consolodateAffiliation, 2))

# HOW TO COUNT AFFILIATIONS
print(results2.where("Affiliation", are.equal_to("Cognitive Science")).num_rows)
print(results2.where("Affiliation", are.equal_to("Psychology")).num_rows)
print(results2.where("Affiliation", are.equal_to("Philosophy")).num_rows)
print(results2.where("Affiliation", are.equal_to("Artificial Intelligence")).num_rows)
print(results2.where("Affiliation", are.equal_to("Anthropology")).num_rows)
print(results2.where("Affiliation", are.equal_to("Neuroscience")).num_rows)
print(results2.where("Affiliation", are.equal_to("Linguistics")).num_rows)
print(results2.where("Affiliation", are.equal_to("Social Science")).num_rows)
print(results2.where("Affiliation", are.equal_to("Other")).num_rows)
#results2

results2.where("Year", are.equal_to(1980)).sort("Article Name").show()

keys = np.unique(results2['Affiliation'])
counts = np.zeros(len(keys))
print(keys)
for i,k in enumerate(keys):
    counts[i] = results2.where("Affiliation", are.equal_to(k)).num_rows

plt.figure(figsize=(10,10))
plt.bar(range(len(counts)),counts)
plt.xticks(range(len(counts)), keys, rotation=45);
#plt.ylim([0,500])

# Artificial Intelligence
results2.where("Affiliation", are.equal_to("Artificial Intelligence")).hist("Year")

# Psychology
results2.where("Affiliation", are.equal_to("Psychology")).hist("Year")

# Philosophy
results2.where("Affiliation", are.equal_to("Philosophy")).hist("Year")

results2.where("Affiliation", are.equal_to("Philosophy")).show()

# Linguistics
results2.where("Affiliation", are.equal_to("Linguistics")).hist("Year")

# Anthropology
results2.where("Affiliation", are.equal_to("Anthropology")).hist("Year")

results2.where("Affiliation", are.equal_to("Anthropology")).show()

# Neuroscience
results2.where("Affiliation", are.equal_to("Neuroscience")).hist("Year")

# Logic, Language and Computation,
# Physics
# Decision technologies
# Neural computation
# Psycholinguistics Psycholinguistique
# Kinesiology
# department of speech pathology
# US ARMY Aeromedical Research
# Asian Studies
# Behaviorial Science
# Communication
# biophysics
# CECS
# Engineering
# Language Sciences
# Neural Science
# Scienza Cognitiva
# Department of Accounting and MIS
# Biological Sciences \n biology
# Electrical and Electronic Engineering
# Department of English Philology
# Department of Environmental Information
# Department of Hospital Medicine
# Department of Human Development
# Department of Humanities
# Department of Management Science and Information Systems
# Department of Management and Economics
# Department of Management and Organization
# Department of Mathematics
# Department of Organismic and Evolutionary Biology
# Department of Physical Medicine and Rehabilitation
# Department of Physical Therapy, University of Connecticut
# Department of Physiology, Feinberg School of Medicine
# Department of Psychiatry and the Behavioral Sciences
# Department of Radiology
# Department of Teaching, Learning, & Culture
# Discipline of Pharmacology
# Asian and African Area Studies

#classify departments into multiple bins
others = results2.where("Affiliation", are.equal_to("Other"))
others.select(2).sort(0,distinct=True).num_rows

# All below cells are tests to see what the data is doing

df_dept = pd.read_csv("cleaned_data")
colnames= ["Article Id","Author","Department_University","Corresponding Author","Article Name","Type"]
df_dept_new=pd.read_csv("cleaned_data", names=colnames)

bad_inds = np.where(df_dept_new['Department_University'].isnull())[0]
len(bad_inds)
#len(df_dept_new)

df_pd.iloc[368]['authors_and_details']

df_dept_new.iloc[bad_inds]

pd.set_option('display.max_rows', 100)
df_dept_new.groupby('Department_University',sort=True).Department_University.count().to_frame().sort_values('Department_University',ascending=False)

# index the row (e.g. 2nd) and split by /n character
df['authors_and_details'][2].split('/n')

# the 0th element contains the information of the first author
df['authors_and_details'][2].split('/n')[0]

# further split author info to separate name and affiliation
au1 = df['authors_and_details'][2].split('/n')[0].split('             ')

au1

# find the corresponding author and delete that part of the string ',Corresponding author'
if 'Corresponding author' in au1[0]:
    endind = au1[0].find(',Corresponding author')
    print(endind)
    print(au1[0][:endind], '- ', au1[1])

# make a new dataset

df['authors_and_details'][1501].split('/n')

df['article_name'][1501]

