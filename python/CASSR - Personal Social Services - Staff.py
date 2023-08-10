#Set this to find the info for your council
councilCrib='Wight'

#Original data file
#http://www.content.digital.nhs.uk/catalogue/PUB23291

import pandas as pd

#fn='http://www.content.digital.nhs.uk/catalogue/PUB23291/pss-staff-eng-16-tables.xlsx'

#For using a local copy...
#!mkdir -p data/
#!wget -P data/ http://www.content.digital.nhs.uk/catalogue/PUB23291/pss-staff-eng-16-tables.xlsx
fn='data/pss-staff-eng-16-tables.xlsx'

xl=pd.read_excel(fn,sheetname=None)
xl.keys()

cover=pd.read_excel(fn,sheetname='Cover').dropna(how='all').dropna(how='all',axis=1)
cover.iloc[14:35,0].tolist()

#Use full table name to generate sheet name and associate the two
cassrSheets=[(c,c.split(':')[0].replace('able ','')) for c in cover.iloc[:,0] if 'CASSR' in c]
cassrSheets

def sheetCleaner(df):
    #The spreadsheet has the data midway down the sheet with multilevel column headings
    #Identify the lower column heading row
    cols=df[df.iloc[:,0]=='ONS Code']
    #Take all the data in the sheet down from the top multilevel heading
    df2=df.iloc[cols.index[0]-1:,:].reset_index(drop=True)
    #Set the top left column heading (currently blank)
    df2.iloc[0,0]='Base'
    #Fill across on top level column headings
    df2.iloc[0,:].fillna(method='ffill',inplace=True)
    #Set and clean the column headings - remove excessive whitespace and strip
    df2.columns=[c.str.replace(r'\W+',' ').str.strip() for c in [df2.iloc[0,:],df2.iloc[1,:]]]
    #The actual data is the datatable less the column headings...
    df2=df2[2:]
    #clean non-data rows at end
    df2.dropna(subset=[('Base','Region')],inplace=True)
    return df2

def df_grabber(fn,sheet):
    #Read in a particular sheet from the spreadsheet
    df=pd.read_excel(fn, sheetname=sheet,
                     na_values=['*','-']).dropna(how='all').dropna(axis=1,how='all').reset_index(drop=True)
    #Return a cleaned dataframe version of the data in that sheet
    return sheetCleaner(df)

df2=df_grabber(fn,'T3a')
df2[df2['Base','Council Name'].str.contains('Wight')]

df2=df_grabber(fn,'T6c(ii)')
df2[df2['Base','Council Name'].str.contains('Wight')]

admincols=['ONS Code', 'Region', 'Council Code', 'Council Name']

def subRowReport(group):
    nulls=[]
    for k in [c for c in group.index if c not in admincols]:
        if not pd.isnull(group.ix[k]):
            print('\t\t{}: {}'.format(k,group.ix[k]))
        else:
            nulls.append(k)
    if nulls: print('\t\tZero/unreported counts for: {}'.format(', '.join(nulls)))
    
def rowReport(row,sheetname):
    print('Report on {} ({}) for {} ({}, {})'.format(sheetname.split(':')[1].strip(),sheetname.split(':')[0].strip(),
                                                     row['Base','Council Name'],row['Base','ONS Code'],
                                                    row['Base','Region']))

from collections import OrderedDict

#Linearise the reports
for sheetname, sheet in cassrSheets:
    df2=df_grabber(fn,sheet)
    df2=df2[df2['Base','Council Name'].str.contains(councilCrib)]
    df2.apply(rowReport,sheetname=sheetname,axis=1)

    #Within each sheet, go through the top-level heading, and then do a subreport of cols in it
    #There's probably a pandas idiomatic way of doing this? Treat levels as groups?
    #For now, fudge it - create an ordered dict to represent: toplevelHeading:sublevelHeading
    #The ordering preserves the column order from the spreadsheet
    coldict=OrderedDict()
    for col in df2.columns:
        if col[0] not in coldict:
            coldict[col[0]]=[]
        coldict[col[0]].append(col[1])
    #Now we can iterate through the toplevel headings
    for k in coldict:
        #Ignoring the admin data stuff if there is only admin data
        if k=='Base':
            nonadmincols=[c for c in df2.xs('Base', level=0, axis=1).columns if c not in admincols]
            if not nonadmincols: continue
        print('\n\t{}'.format(k))
        df2.xs(k, level=0, axis=1).apply(subRowReport,axis=1)
        
    print('\n--------\n')
#'ONS Code','Region','Council Code','Council Name'

tmp={'All council job roles':1,'Direct Care':2,'Managers Supervisor':3,'Regulated Professions':4,'Other':5}

ll='''
There were {All council job roles} job roles across the council, \
of which {Direct Care} were associated with direct care, \
{Managers Supervisor} were managerial/supervisory roles, \
{Regulated Professions} were in regulated professions and {Other} other.
'''.format(**tmp)
print(ll)

#Can we do nested dicts?
tmp={'a':{'b a':1,'c':{'c c':3},'d':4},'e':0}
print('{a[b a]}, {a[c][c c]}'.format(**tmp))

#but the iniial key eg 'a' does not support spaces?

sheet='T6a(ii)'
ss=df_grabber(fn,sheet)
ss=ss[ss['Base','Council Name'].str.contains(councilCrib)]

print(ss.columns)
ss

#Write the following template using paths to the appropriate table, top level headings and subheadings


l2='''
For *{r[T6a(ii)][name]}* ({r[T6a(ii)][key]}), across all job roles there were {r[T6a(ii)][All Job Roles][Male]} males and {r[T6a(ii)][All Job Roles][Female]} females \
({r[T6a(ii)][All Job Roles][Not recorded Unknown]} not recorded or unknown).

This breaks down as follows:

- in direct care, {r[T6a(ii)][Direct Care][Male]} males and {r[T6a(ii)][Direct Care][Female]} females \
({r[T6a(ii)][Direct Care][Not recorded Unknown]} not recorded or unknown)

- in managerial or supervisory roles, {r[T6a(ii)][Managers Supervisor][Male]} males and {r[T6a(ii)][Managers Supervisor][Female]} females \
({r[T6a(ii)][Managers Supervisor][Not recorded Unknown]} not recorded or unknown)

- amongst the regulated professions, {r[T6a(ii)][Regulated Professions][Male]} males and {r[T6a(ii)][Regulated Professions][Female]} females \
({r[T6a(ii)][Regulated Professions][Not recorded Unknown]} not recorded or unknown)

- and for other roles, {r[T6a(ii)][Other][Male]} males and {r[T6a(ii)][Other][Female]} females \
({r[T6a(ii)][Other][Not recorded Unknown]} not recorded or unknown)
'''

#This code churns a sheet into a dict that we can apply to the template.
cc={'r':{sheet:{'key':sheet, 'name':[k for k,v in cassrSheets if v==sheet][0]}}}

coldict=OrderedDict()
for col in ss.columns:
    if col[0] not in coldict:
        coldict[col[0]]=[]
    coldict[col[0]].append(col[1])

bb={}
#Now we can iterate through the toplevel headings
for k in coldict:
    #Ignoring the admin data stuff if there is only admin data
    if k=='Base':
        nonadmincols=[c for c in df2.xs('Base', level=0, axis=1).columns if c not in admincols]
        if not nonadmincols: continue
    bb[k]=ss[k].fillna(0).to_dict(orient='records')
    for v in bb:
        cc['r'][sheet][v]=bb[v][0]

        
#Now we can apply the dict obtained from the churned sheet to the template
print(l2.format(**cc))

l2._meta.get_all_field_names()



