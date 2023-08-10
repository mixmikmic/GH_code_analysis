import pandas as pd
import os
import openpyxl as px
import re 

cwd = os.getcwd()
datadir = '/'.join(cwd.split('/')[0:-1]) + '/Raw Data/'
df = pd.read_excel(datadir+'/judges.xlsx', sheetname="judges")
df['law school']= df['law school'].fillna('Unknown')
df['re-elect years'] = df['re-elect years'].fillna('None')

df['law school'] = df['law school'].apply(lambda x:re.sub(' ','',x.strip()).lower())
#Get distinct name list 
list_law_school = df['law school'].unique()

list_electyear = df['re-elect years'].unique()

list_electyear

list_electyear_unique = []
for e in list_electyear:
    if ';' in str(e):
        c = e.count(';')
        for each in e.split(';',c):
            if '?' in each:
                each = each.split('?')[0]
            list_electyear_unique.append(each)
    elif ',' in str(e):
        c = e.count(',')
        for each in e.split(',',c):
            list_electyear_unique.append(each)
    elif 'x' or 'None' in str(e):
        continue
    elif '?' in str(e):
        list_electyear_unique.append(e.split('?')[0])
    else:
        list_electyear_unique.append(str(e))

#remove white space
list_electyear_unique = [item.strip() for item in list_electyear_unique]
list_electyear_unique_final = list(set(list_electyear_unique))


         

list_electyear_unique_final.sort()
del list_electyear_unique_final[0]

for year in list_electyear_unique_final:
    df[year] = df['re-elect years'].apply(lambda x: 1 if (year in str(x) and '?' not in str(x)) else 0)
del df['re-elect years']

df = df.drop([658,659,660,661])
df['grad year'] = df['grad year'].apply(lambda x: x.split(',')[0] if ',' in str(x) else x)
df['notes'].unique()

notes_column = ['retired','arrested','lost','died','left','appointed','contested']
for notes in notes_column:
    df[notes] = df['notes'].apply(lambda x: 1 if notes in str(x) else 0)
df = df.rename(columns={'lost': 'lost-reelection', 'appointed': 'appointed-to-other-depart'})
del df['notes']

delete_columns = ['Unnamed: 28','web','lower only','no experience info']
for de in delete_columns:
    del df[de]

remove_list = ['missing','x','?',0]
df['start year'] = df['start year'].apply(lambda x:'Unknown' if (x in remove_list or '?' in str(x)) else x)

df_1 = df[df['multiple'] != 1]
df_2 = df[df['multiple'] == 1]

columns_to_delete_1 = ['judges (new)', 'Unnamed: 1','      Freq.     Percent        Cum.','multiple','misspell','missing']
for col in columns_to_delete_1:
    del df_1[col]
columns_to_delete_2 = ['      Freq.     Percent        Cum.','multiple','misspell','missing']
for col in columns_to_delete_2:
    del df_2[col]

df_2 = df_2.rename(columns={'judges (original)': 'judges_last_name', 'Unnamed: 3': 'judges_first_name'})

df_1 = df_1.rename(columns={'judges (original)': 'judges_last_name', 'Unnamed: 3': 'judges_first_name'})

del df_2['Unnamed: 1']
del df_2['judges (new)']

df_final = df_1.append(df_2)

df_final

df_final.to_csv('judges_final.csv')

#read sheet from excel file
df_appellate = pd.read_excel(datadir+'/judges.xlsx', sheetname="appellate")

#reset index and change column name for last name
df_appellate = df_appellate.reset_index()
df_appellate = df_appellate.rename(columns={'index':'Appellate_Last_Name'})

#clean law school based on same format as judges
df_appellate['law school']= df_appellate['law school'].fillna('Unknown')
df_appellate['law school'] = df_appellate['law school'].apply(lambda x:re.sub(' ','',x.strip()).lower())

df_appellate['law school'].unique()

df_appellate['grad year'] = df_appellate['grad year'].apply(lambda x: x if pd.isnull(x) else int(x))

list_electyear_appellate = df_appellate['re-elect years'].unique()

list_electyear_unique = []
for e in list_electyear_appellate:
    if ',' in str(e):
        c = e.count(',')
        for each in e.split(',',c):
            list_electyear_unique.append(each)
    elif 'x' in str(e):
        continue
    elif '?' in str(e):
        list_electyear_unique.append(e.split('?')[0])
    else:
        list_electyear_unique.append(str(e))

list_electyear_unique_2 = []
for e in list_electyear_unique:
    if '?' in str(e):
        list_electyear_unique_2.append(e.split('?')[0])
    else:
        list_electyear_unique_2.append(str(e))
        
    
#remove white space
list_electyear_unique_2 = [item.strip() for item in list_electyear_unique_2]
list_electyear_unique_final = list(set(list_electyear_unique_2))

for year in list_electyear_unique_final:
    df_appellate[year] = df_appellate['re-elect years'].apply(lambda x: 1 if (year in str(x) and '?' not in str(x)) else 0)

del df_appellate['re-elect years']

#deal with notes column
df_appellate['notes'].unique()

notes_column = ['retired','presiding justice']
for notes in notes_column:
    df_appellate[notes] = df_appellate['notes'].apply(lambda x: 1 if notes in str(x) else 0)

del df_appellate['notes']

df_appellate

df_appellate.to_csv('appellate_final.csv')



