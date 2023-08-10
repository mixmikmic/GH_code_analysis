import pandas as pd
import numpy as np

da = pd.read_csv('updated_DA.csv')

da.head()

main = pd.read_csv('../Raw Data/raw_criminal.csv')

main.head()

main.columns

#Names in the main file
main_DA_names = main.DistrictAttorney.unique()
main_DA_names = [x for x in main_DA_names if str(x) != 'nan']
main_DA_names

#Extract last name and convert to lower case

def extract_last_name(names):
    '''
    Extract lower case last name from a full name
    '''
    judges_last_names = {}
    judges = []
    
    for name in names:
        name = name.replace(', Special','')
        name = name.replace('[*2]','' )
        name = name.replace('—[*1]','' )
        name = name.strip() #remove trailing spaces
        separated = name.split(' ')
        if len(separated)>2:
            last_name = separated[2]
        elif len(separated)==2:
            last_name = separated[1]
        else:
            print(separated)
        last_name = last_name.replace(',','')
        judges_last_names[name] = last_name.lower()
        judges.append(last_name.lower())
    return judges_last_names,judges

judges_last_names,judges = extract_last_name(main_DA_names)

judges

#Names in the DA file
DA_last_names = da.da_last_name.unique()

#Make a list of matching names for later merging
out_name = {}
not_found = []

for judge_name in judges_last_names.keys():
    found = False
    for last_name in DA_last_names:
        try:
            if last_name.lower() == judges_last_names[judge_name]:
                out_name[judge_name] = last_name
                found = True
                continue

        except (TypeError, AttributeError):
            continue
            
    if not found:
        not_found.append(judge_name)

out_name

#Unique names not found
set(not_found)

#Find similar names from judges file
from difflib import get_close_matches, SequenceMatcher

lower_case_da = list(map(lambda x: x.lower(), DA_last_names))
missing = []

for name in not_found:
    judge_last_name = judges_last_names[name]
    try:
        best_match = get_close_matches(judge_last_name, lower_case_da)[0]
        score = SequenceMatcher(None, judge_last_name, best_match).ratio()
        
        if score > 0.82: #if match is close enough
            out_name[name] = last_name
            print(name, best_match, score)
        
        else:
            missing.append(name)
            print('missing:', name, judge_last_name)
               
    except IndexError:
        missing.append(name)

#Missing DA names
missing

missing_da_counties = main[main['DistrictAttorney'].isin(missing)].County

updated_main = pd.read_csv('criminal_main.csv',encoding='latin-1')

updated_main.head()

updated_main.columns

temp = pd.DataFrame([updated_main['File'], updated_main['Appeal_Date']]).transpose()
#temp

#Add year column to main dataframe
main = pd.merge(main,temp, how='left', on='File')

def string_to_year(s):
    try:
        return int(s[0:4])
    except TypeError:
        return ''

main['appeal_year'] = list(main.Appeal_Date.map(lambda x: string_to_year(x)))

missing_da_years = main[main['DistrictAttorney'].isin(missing)].appeal_year

missing[0]

main[main.DistrictAttorney == missing[1]]

main.columns

#Add existing DA's to main frame
main['DA_last_name'] = main.DistrictAttorney.map(lambda x: out_name.get(x,'Unknown'))

main.head()


names = main[(main['appeal_year']==2012) & (main['County']=='Cayuga')].DistrictAttorney

index_d = {}

for name in missing:
    subset = main[main.DistrictAttorney == name]
 
    if isinstance(subset,pd.DataFrame):
        for ind, subname in subset.iterrows():
            year = subname.appeal_year
            county = subname.County
            names = main[(main['appeal_year']==year) & (main['County']==county)].DA_last_name
            print(names)
            current_name = subname.DistrictAttorney
            #names = names.replace(current_name,None)
            names = names.mode()
            val = names.tolist()
            index_d[current_name] = val
        
    else:
        year = subset.appeal_year
        county = subset.County
        names = main[(main['appeal_year']==year) & (main['County']==county)].DA_last_name
        index_d[name] = names     
            

index_d

final_missing = []
for key in index_d.keys():
    val = index_d[key]
    if val[0]=='Unknown':
        final_missing.append(key)
    else:
        out_name[key]=val

print(len(missing))

#Final people I need to check manually
final_missing

#Looks like a number of DA's are missing from the file
out_name['Irene C. Graven'] = 'Graven'
out_name['James A. McCarty'] = 'McCarty'
out_name['Robert Tendy'] = 'Tendy'
out_name['Kristyna S. Mills'] = 'Mills'
out_name['Michael J. Flaherty, Jr.'] = 'Flaherty'
out_name['Patrick E. Swanson'] = 'Swanson'
out_name['Eric Gonzalez'] = 'Gonzalez'
out_name['Anthony A. Scarpino, Jr.'] = 'Scarpino'
out_name['Kelli P. McCoski'] = 'McCoski'
out_name['Theodore A. Brenner, Deputy'] = 'Brenner'
out_name['Theodore A. Brenner, Deputy'] = 'Brenner'
out_name['John J. Flynn'] = 'Flynn'
out_name['Patrick A. Perfetti'] = 'Perfetti'
out_name['Caroline A. Wojtaszek'] = 'Wojtaszek'
out_name['Matthew Van Houten'] = 'Van Houten'
out_name['Matthew VanHouten'] = 'Van Houten'
out_name['Eliza Filipowski'] = 'Filipowski'

out_name

import csv
#Write key table to csv for matching files later
with open('da_last_name_dictionary.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in out_name.items():
        if isinstance(val, list):
            val = val[0]
        writer.writerow([key, value])

