import pandas as pd

df = pd.read_csv('../Ksenia/da_mode_grounds_data_add_crime_unanimous_appellant_dummies_dedup.csv',encoding='latin-1')

df.head()

df['File'].duplicated().any()

df.Court.unique()

df['Court'] = df['Court'].apply(lambda x: '_'.join([i.lower().title() for i in str(x).split(' ') if x != 'nan']))

df['Court'] = df['Court'].apply(lambda x:'Missing' if x == 'Nan' else x)

df['County'] = df['County'].apply(lambda x: 'Missing' if x == 'by County' else x)
df['County'] = df['County'].apply(lambda x: 'Missing' if x == 'Unknown' else x)
df['County'] = df['County'].apply(lambda x: 'Westchester County' if x == 'Weschester County' else x)
df['County'] = df['County'].fillna('Missing')

df['County'].unique()

df_new = df[['File','Court','County']]

dummies = pd.get_dummies(df_new[['Court','County']])

df_final = pd.concat([df['File'], dummies], axis=1)

df_final.columns

df_final.to_csv(r'csv/Cleaned_Court_County.csv')

df_judge = df[['File','Judge']]

judge_info = pd.read_csv(r'csv/judges_final.csv')

df_judge['Judge_New'] = df_judge['Judge'].fillna('Missing')

df_judge['Judge_New'] = df_judge['Judge_New'].apply(lambda x: x.split(',')[0] if ',' in x else x)

def Clean_Last_Name(data):

    if '.' in data:
        result = data.split('.')[-1]
    elif ' ' in data:
        result = data.split(' ')[-1]
    else:
        result = data
    return result

def Clean_First_Name(data):
    if len(data.split(' ')) > 1:
        result = data.split(' ')[0]
        if '.' in result:
            result = data.split(' ')[1]
    else:
        result = 'Missing'
    return result

df_judge['Last_Name'] = df_judge['Judge_New'].apply(lambda x:Clean_Last_Name(x))

df_judge['Last_Name'] = df_judge['Judge_New'].apply(lambda x:x.split(' ')[-1] if len(x.split(' '))!=1 else x)

df_judge['First_Name'] = df_judge['Judge_New'].apply(lambda x:Clean_First_Name(x))

df_judge['Last_Name'] = df_judge['Last_Name'].apply(lambda x:x.lower())
df_judge['First_Name'] = df_judge['First_Name'].apply(lambda x:x.lower())

df_judge_final = df_judge[['File','First_Name','Last_Name']]

df_judge_final['First_Name'].isnull().any()

df_judge_final['First_Name'] = df_judge_final['First_Name'].apply(lambda x:x.strip())

df_judge_final['Last_Name'].isnull().any()

df_judge_final['Last_Name'] = df_judge_final['Last_Name'].apply(lambda x:x.strip())

df_judge_final.head()

df_judge_final.to_csv(r'csv/judge_index.csv')

judges_distinct_last_name = judge_info[judge_info['judges_first_name'].isnull()]
judges_same_last_name = judge_info[~judge_info['judges_first_name'].isnull()]

#get judge_info without first_name. Distinct last name
judges_distinct_last_name.head()

df_judge_final['Last_Name'] = df_judge_final['Last_Name'].apply(lambda x: x.strip())
judges_distinct_last_name['judges_last_name'] = judges_distinct_last_name['judges_last_name'].apply(lambda x: x.strip())

merge_1 = df_judge_final.merge(judges_distinct_last_name,left_on='Last_Name',right_on='judges_last_name',how = 'left')

merge = merge_1.merge(judges_same_last_name,left_on=['First_Name','Last_Name'],right_on=['judges_first_name','judges_last_name'],how = 'left')

for i in merge.columns:
    if 'y' in i:
        del merge[i]

del merge['judges_first_name_x']

del merge['ADA_x']

merge = merge.rename(columns = {'First_Name':'Judge_First_Name','Last_Name':'Judge_Last_Name','law school_x':'Judge_Law_School'             , 'elect_x':'Judge_Elect','APD_x':'Judge_APD','Prof_x':'Judge_Prof','female_x':'Judge_Female',             'CC_x':'Judge_CC','SC-AJ_x':'Judge_SC-AJ','SC_x':'Judge_SC','Judge_SC-AJ & SC_x':'Judge_SC-AJ & SC',              'ballotpedia_x':'Judge_Ballotpedia', '1976_x':'Judge_Elect_1976', '1980_x':'Judge_Elect_1980',             '1981_x':'Judge_Elect_1981', '1984_x':'Judge_Elect_1984','1985_x':'Judge_Elect_1985', '1986_x':'Judge_Elect_1986'               , '1988_x':'Judge_Elect_1988', '1989_x':'Judge_Elect_1989', '1990_x':'Judge_Elect_1990', '1991_x':'Judge_Elect_1991',               '1992_x':'Judge_Elect_1992','1993_x':'Judge_Elect_1993', '1994_x':'Judge_Elect_1994', '1995_x':'Judge_Elect_1995',               '1996_x':'Judge_Elect_1996', '1997_x':'Judge_Elect_1997', '1998_x':'Judge_Elect_1998', '1999_x':'Judge_Elect_1999',               '2000_x':'Judge_Elect_2000', '2001_x':'Judge_Elect_2001', '2002_x':'Judge_Elect_2002', '2003_x':'Judge_Elect_2003',               '2004_x':'Judge_Elect_2004', '2005_x':'Judge_Elect_2005', '2006_x':'Judge_Elect_2006', '2007_x':'Judge_Elect_2007',               '2008_x':'Judge_Elect_2008', '2009_x':'Judge_Elect_2009', '2010_x':'Judge_Elect_2010', '2011_x':'Judge_Elect_2011',               '2012_x':'Judge_Elect_2012', '2013_x':'Judge_Elect_2013', '2014_x':'Judge_Elect_2014', '2015_x':'Judge_Elect_2015',               '2016_x':'Judge_Elect_2016', 'retired_x':'Judge_Retired', 'arrested_x':'Judge_Arrested','lost-reelection_x':'Judge_Lost_Reelection',               'died_x': 'Judge_Died', 'left_x':'Judge_Left', 'appointed-to-other-depart_x':'Judge_Appointed_To_Other_Department',               'contested_x':'Judge_Contested'})

merge.columns

merge['Judge_Law_School'] = merge['Judge_Law_School'].apply(lambda x:'missing' if x == 'unknown' else x)

merge = merge.fillna('missing')

merge

merge.to_csv('merged_with_judges.csv')

df['File'].duplicated().any()



