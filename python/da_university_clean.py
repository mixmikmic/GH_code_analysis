import pandas as pd

judge_universities = ['cuny', 'columbia', 'tulsa', 'syracuse', 
                      'brooklyn', 'sanfrancisco','unknown', 'pace', 
                      'cornell', 'akron', 'st.johns', 'nyu', 'fordham',
                      'widener', 'albany', 'gonzaga', 'touro', 
                      'buffalo', 'casewestern','suffolk', 'hofstra',
                      'harvard', 'westernnewengland', 'georgetown',
                      'cardoza', 'yale', 'georgewashington', 
                      'baltimore', 'colorado','delaware', 'toledo', 
                      'temple', 'villanova', 'rutgers',
                      'californiawestern', 'bostonu', 'michigan', 
                      'franklinpierce','newengland', 'stjohns', 
                      'capital', 'pugetsound', 'ohiostate',
                      'pittsburgh', 'william&mary', 'dayton', 
                      'johnmarshall', 'duke','vermont', 'penn', 
                      'kent', 'loyola', 'novase', 'bostonc', 
                      'boalt','catholic', 'valparaiso', 'seattle', 
                      'washington', 'st.thomas','maryland', 
                      'duquesne', 'memphis', 'clevelandst.', 
                      'howard','michiganst.', 'ohionorthern', 
                      'kentst.', 'washburn','thomascooley', 'emory', 
                      'virginia', 'clevelandmarshall','puertorico', 
                      'cal-hastings', 'antioch', 'dickenson', 
                      'georgia','hamline', 'johnjay', 'southtexas', 
                      'bostoncollege', 'wisconsin']

da_data = pd.read_csv('clean_DA.csv')

da_data.columns

da_universities = da_data['university']

#Skip empty rows - missing university names
da_universities.dropna(inplace=True)

da_universities = da_universities.unique()

da_uni = list(entry.lower() for entry in da_universities)

da_uni

#Remove spaces
univ = []

for uni in da_uni:
    ex_empty = "".join(uni.split())
    univ.append(ex_empty)

university_dict = {}
missing = []

for school in univ:
    if school in judge_universities:
        university_dict[school] = school
    else:
        missing.append(school)

missing

#Find similar names from judges file
from difflib import get_close_matches

for school in missing:
    possibilities = get_close_matches(school, judge_universities)
    print(school, possibilities)

#I checked that Penn refers to University of Pennsylvania here
university_dict['universityofpennsylvania'] = 'penn'

missing.remove('universityofpennsylvania')

#update dictionary with missing calues
for value in missing:
    university_dict[value] = value 

#Update da_universities list
universities = []
da_uni=da_data['university']

for college in da_uni:
    try: #if not empty
        
        college = college.lower()
        college = "".join(college.split())
    except AttributeError:
        college = 'unknown'
    universities.append(college)

#Create a list to match DA universities
final_list = []

for college in universities:
    try:
        name = university_dict[college]
    except KeyError:
        name = 'unknown'
    final_list.append(name)

#Add updated university names to dataframe
da_data.drop('university',axis=1, inplace=True)

da_data['university'] = final_list

da_data.head()

da_data.to_csv('clean_DA.csv')

