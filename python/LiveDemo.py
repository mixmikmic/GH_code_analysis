import pandas as pd
import numpy as np
import re
get_ipython().run_line_magic('load_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-iv -v -d')

excelfile = pd.ExcelFile('./terrible_spreadsheet.xlsx')

firstsheet = excelfile.sheet_names[0]
excelfile.sheet_names

ff = pd.read_excel(excelfile, sheetname=firstsheet, header=1)
ff.shape

df = pd.concat([pd.read_excel(excelfile, sheetname=sheet, header=1).assign(sheet=sheet)
                for sheet in excelfile.sheet_names])

df.shape

df.columns

oldcols = df.columns

colcleaningregex = re.compile(r'[^\w]')

newcols = [colcleaningregex.sub('_', col.strip()) for col in df.columns]

print(len(oldcols)-len(oldcols.unique()),
     len(newcols)-len(np.unique(newcols)))

df.columns = newcols

df.notnull().any()

df = df.loc[:, df.notnull().any()]

df = df.reset_index(drop=True) #required in order for the groupby-apply to work
df.loc[:, ['Hospital', 'Age', 'Gender']] = df.groupby('sheet').apply(
    lambda x: x.loc[:, ['Hospital', 'Age', 'Gender']].fillna(method='ffill')
)

df.loc[:, ['Hospital', 'Age', 'Gender']].iloc[10:15]

get_ipython().run_line_magic('pinfo', 'df.Sample_ID.str')

#df.Sample_ID.str.extract(r'.*\s(?P<Dilution>1[0]+)', expand=True)
df.Sample_ID.str.extract(r'\s(?P<Visit>[^\s]*)\s', expand=False).unique()

results = df.Sample_ID.str.extract(
    r'(?P<PatientID>\d{5})?\s*(?P<Visit>[^\s]+\d)?\s+(?P<Dilution>1[0]+)?\s*$', 
                         expand=True)
results

results.Visit.unique()

results.PatientID.unique()

results.Dilution.unique()



