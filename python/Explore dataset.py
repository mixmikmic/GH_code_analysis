import pandas as pd
import numpy as np

# Unique value
def unique_value(df):
    uvlist = []
    col_list = list(df.columns)
    for i in col_list:
        uvlist.append(len(df[i].unique()))
    return uvlist

# Vacancy Rate
def vacancy_rate(df):
    vrlist = []
    col_list = list(df.columns)
    for i in col_list:
        vrlist.append(len(df[df[i].isnull()])/float(len(df)))
    return vrlist

# Unique Sample
def unique_sample(df):
    uslist = []
    col_list = list(df.columns)    
    for i in col_list:
        uslist.append(list(df[i].unique())[:5])
    return uslist

# Value Count
def value_count(df, topnumber):
    vclist = []
    top = topnumber-1
    col_list = list(df.columns)    
    for i in col_list:
        try:
            vclist.append((df[i].value_counts().index[top], df[i].value_counts()[top]))
        except:
            vclist.append(None)
    return vclist

def infor_table(df):
    table = [unique_value(df),vacancy_rate(df),unique_sample(df),value_count(df,1),value_count(df,2),value_count(df,3)]
    infodf = pd.DataFrame(table)
    infodf = infodf.transpose()
    cols = ['unique_value', 'vacancy_rate','5_sample','value_count_top1','value_count_top2','value_count_top3']
    infodf.columns = cols
    col_list = list(df.columns)
    infodf.index = col_list
    return infodf

def explore(PATH):
    Filename = PATH
    df = pd.read_csv(Filename)
    df.info()
    return infor_table(df)   

explore('Consumer_Complaints_with_Consumer_Complaint_Narratives.csv')

