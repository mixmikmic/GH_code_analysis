import pandas as pd
def to_xlsx(filename):
    return "{}.xlsx".format(filename)
def to_xls(filename):
    return "{}.xls".format(filename)

get_ipython().magic("time xl = pd.ExcelFile(to_xlsx('Escalade2015'))")
xl.sheet_names

df = xl.parse('Sheet1', usecols=['pays', 'année', 'rang', 'temps net', 'catégorie'])#, 'nom'])
df.head()

from math import nan
def get_age(year):
    try:
        return 2015 - int(year)
    except:
        return nan

df['Age'] = df['année'].apply(get_age)

def get_vitesse(temps):
    try:
        temps = float(temps.replace(',',''))
        return (60/temps)*7.323
    except:
        return nan

get_vitesse(23)

catlist = set(df['catégorie'].tolist())
catlist

df['vitesse'] = df['temps net'].apply(get_vitesse)

def selCategory(category):
#    if category in ('Escaladélite femmes', 'Escaladélite hommes', 'Femmes I', 'Femmes II', 'Femmes III', 'Femmes IV', 'Femmes V', 'Femmes VI', 'Hommes I', 'Hommes II', 'Hommes III', 'Hommes IV', 'Hommes V', 'Hommes VI'):
    if category in ('Escaladelite Femmes', 'Escaladelite Hommes', 'Femmes I', 'Femmes II', 'Femmes III', 'Femmes IV', 'Femmes V', 'Femmes VI', 'Hommes I', 'Hommes II', 'Hommes III', 'Hommes IV', 'Hommes V', 'Hommes VI'):
        return True
    else:
        return False
dfx = df[df['catégorie'].apply(selCategory)]
dfx.to_csv('Data_selection.csv')





