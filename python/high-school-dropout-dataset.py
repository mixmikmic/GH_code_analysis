get_ipython().run_line_magic('pylab', 'inline')

import re
import numpy as np
import pandas as pd

def cleandata(x):
    if (isinstance(x, str)):
        if re.search("[P][S]", x) and len(x) == 2:
            #z = pd.to_numeric(re.sub('[PS]', '', x))
            y = np.random.randint(0,100)

            return y
        if re.search("[L][T][0-9][0-9]", x) and len(x) == 4:
            z = pd.to_numeric(re.sub('[LT]', '', x))
            y = np.random.randint(0,z)
            return y
        
        if re.search("[L][E][0-9][0-9]", x) and len(x) == 4:
            z = pd.to_numeric(re.sub('[LE]', '', x))
            y = np.random.randint(0,z)
            return y
        
        if re.search("[L][E][0-9]", x) and len(x) == 3:
            z = pd.to_numeric(re.sub('[LE]', '', x))
            y = np.random.randint(0,z)
            return y
        
        if re.search("[G][E][0-9][0-9]", x) and len(x) == 4:
            z = pd.to_numeric(re.sub('[GE]', '', x))
            y = np.random.randint(z,100)

            return y
        
        if re.search("[G][T][0-9][0-9]", x) and len(x) == 4:
            z = pd.to_numeric(re.sub('[GT]', '', x))
            y = np.random.randint(z,100)

            return y
        
        if re.search("[0-9][0-9]-[0-9][0-9]", x) and len(x) == 5:
            z = x.split('-')
            a = pd.to_numeric(z[0])
            b = pd.to_numeric(z[1])
            y = np.random.randint(a,b)
            
            return y
        
        if re.search("[0-9]-[0-9]", x) and len(x) == 3:
            z = x.split('-')
            a = pd.to_numeric(z[0])
            b = pd.to_numeric(z[1])
            y = np.random.randint(a,b)
            
            return y
    return x

def parsecohortdata(x, y):
    # Get the regulatory adjusted cohort graduation rate for all district
    acgr = pd.read_csv(x, dtype=str, na_values={'.','PS'})

    #Add the Year column
    acgr['YEAR'] = y

    #Rename existing coulmns
    if(len(acgr.columns) == 26):
        acgr.columns = ['STATE','FIPCD','LEAID','LEANM','ALLA','ALLP','MAMA','MAMP',
                        'MASA','MASP','MBLA','MBLP','MHIA','MHIP','MTRA','MTRP','MWHA',
                        'MWHP','CWDA','CWDP','ECDA','ECDP','LEPA','LEPP','DATE_CUR','YEAR']

    if(len(acgr.columns) == 27):
        acgr.columns = ['STATE','FIPCD','LEAID','LEANM','ALLA','ALLP','MAMA','MAMP',
                        'MASA','MASP','MBLA','MBLP','MHIA','MHIP','MTRA','MTRP','MWHA',
                        'MWHP','CWDA','CWDP','ECDA','ECDP','LEPA','LEPP','INSERT_DATE',
                        'DATE_CUR','YEAR']
        del acgr['INSERT_DATE']
    
    # Clean data
    acgr = acgr.applymap(cleandata)
    acgr = acgr.fillna(0)
 
    #Convert all column data to numeric data
    for c in acgr.iloc[:, 4:24].columns:
        acgr[c] = pd.to_numeric(acgr[c])
    
    #Round the data to the nearest int
    acgr['ALLG'] = round(acgr['ALLA']*acgr['ALLP']/100, 0).astype(int)
    acgr['MAMG'] = round(acgr['MAMA']*acgr['MAMP']/100, 0).astype(int)
    acgr['MASG'] = round(acgr['MASA']*acgr['MASP']/100, 0).astype(int)
    acgr['MBLG'] = round(acgr['MBLA']*acgr['MBLP']/100, 0).astype(int)
    acgr['MHIG'] = round(acgr['MHIA']*acgr['MHIP']/100, 0).astype(int)
    acgr['MTRG'] = round(acgr['MTRA']*acgr['MTRP']/100, 0).astype(int)
    acgr['MWHG'] = round(acgr['MWHA']*acgr['MWHP']/100, 0).astype(int)
    acgr['CWDG'] = round(acgr['CWDA']*acgr['CWDP']/100, 0).astype(int)
    acgr['ECDG'] = round(acgr['ECDA']*acgr['ECDP']/100, 0).astype(int)
    acgr['LEPG'] = round(acgr['LEPA']*acgr['LEPP']/100, 0).astype(int)
    
    #Round the data to the nearest decimal
    acgr['MAMR'] = round(acgr['MAMA']*100/acgr['ALLA'],2)
    acgr['MASR'] = round(acgr['MASA']*100/acgr['ALLA'],2)
    acgr['MBLR'] = round(acgr['MBLA']*100/acgr['ALLA'],2)
    acgr['MHIR'] = round(acgr['MHIA']*100/acgr['ALLA'],2)
    acgr['MTRR'] = round(acgr['MTRA']*100/acgr['ALLA'],2)
    acgr['MWHR'] = round(acgr['MWHA']*100/acgr['ALLA'],2)
    acgr['CWDR'] = round(acgr['CWDA']*100/acgr['ALLA'],2)
    acgr['ECDR'] = round(acgr['ECDA']*100/acgr['ALLA'],2)
    acgr['LEPR'] = round(acgr['LEPA']*100/acgr['ALLA'],2)
    
    #Delete additional column
    del acgr['STATE'], acgr['LEANM'], acgr['DATE_CUR']
    
    #return the dataframe
    return acgr[['YEAR','FIPCD','LEAID','ALLA','ALLP','ALLG','MAMA','MAMP','MAMG','MASA','MASP','MASG',
                 'MBLA','MBLP','MBLG','MHIA','MHIP','MHIG','MTRA','MTRP','MTRG','MWHA','MWHP','MWHG',
                 'CWDA','CWDP','CWDG','ECDA','ECDP','ECDG','LEPA','LEPP','LEPG','MAMR','MASR','MBLR',
                 'MHIR','MTRR','MWHR','CWDR','ECDR','LEPR'
                ]]

def parsepovertydata(x):
    sipe = pd.read_csv(x, encoding='latin-1', dtype={"State FIPS Code": str, "District ID": str})
    
    #Rename existing coulmns
    sipe.columns = ['STATE', 'FIPST', 'DISTID', 'LEANM', 'TOTP', 'CHLD', 'CHIP']
    
    #Add a new column as LEAID = 'FIPST' + 'DISTID'
    sipe['LEAID'] = sipe['FIPST'] + sipe['DISTID']

    #Create a new Dataframe with selected column
    sipe = sipe[['LEAID', 'TOTP', 'CHLD', 'CHIP']]
    
    #Replace invalid data in the dataframe
    sipe['TOTP'] = sipe['TOTP'].replace(',', '', regex=True)
    sipe['CHLD'] = sipe['CHLD'].replace(',', '', regex=True)
    sipe['CHIP'] = sipe['CHIP'].replace(',', '', regex=True)

    #Convert all column data in the dataframe to numeric data
    sipe['TOTP'] = pd.to_numeric(sipe['TOTP'])
    sipe['CHLD'] = pd.to_numeric(sipe['CHLD'])
    sipe['CHIP'] = pd.to_numeric(sipe['CHIP'])
    
    #Round the numeric data to nearest int
    sipe['CHIR'] = round(sipe['CHLD']/sipe['TOTP']*100, 0)
    sipe['POVR'] = round(sipe['CHIP']/sipe['CHLD']*100, 0)
    sipe = sipe.fillna(0)
    
    #Convert each column to integer
    sipe['CHIR'] = sipe['CHIR'].astype(int)
    sipe['POVR'] = sipe['POVR'].astype(int)
    
    #Return dataframe
    return sipe[['LEAID', 'CHIR', 'POVR']]

def parseassessmentdata(x, year, subject):
    salm = pd.read_csv(x, dtype = str, na_values=['n/a'])
    z = str(year % 2000)
    y = str(year % 2000) + str((year % 2000)+1)
    
    #Add the Year column
    salm['YEAR'] = year
    
    if (subject == 'MATH'):
        s = 'MTH'
    
    if (subject == 'ENGLISH'):
        s = 'RLA'
        
    salm.columns = [x.upper() for x in salm.columns]
    
    #Rename existing coulmns in the regulatory adjusted cohort graduation rate dataframe
    salm.rename(columns={'STNAM':'STATE', 'LEANM' + z:'LEANM', 
                         'ALL_' + s + 'HSNUMVALID_' + y:'ALL' + s + 'A', 
                         'ALL_' + s + 'HSPCTPROF_' + y :'ALL' + s + 'P',
                         'MAM_' + s + 'HSNUMVALID_' + y:'MAM' + s + 'A', 
                         'MAM_' + s + 'HSPCTPROF_' + y :'MAM' + s + 'P',
                         'MAS_' + s + 'HSNUMVALID_' + y:'MAS' + s + 'A', 
                         'MAS_' + s + 'HSPCTPROF_' + y :'MAS' + s + 'P',
                         'MBL_' + s + 'HSNUMVALID_' + y:'MBL' + s + 'A', 
                         'MBL_' + s + 'HSPCTPROF_' + y :'MBL' + s + 'P',
                         'MHI_' + s + 'HSNUMVALID_' + y:'MHI' + s + 'A', 
                         'MHI_' + s + 'HSPCTPROF_' + y :'MHI' + s + 'P',
                         'MTR_' + s + 'HSNUMVALID_' + y:'MTR' + s + 'A', 
                         'MTR_' + s + 'HSPCTPROF_' + y :'MTR' + s + 'P',
                         'MWH_' + s + 'HSNUMVALID_' + y:'MWH' + s + 'A', 
                         'MWH_' + s + 'HSPCTPROF_' + y :'MWH' + s + 'P',
                         'F_' + s + 'HSNUMVALID_' + y  :'F' + s + 'A', 
                         'F_' + s + 'HSPCTPROF_' + y   :'F' + s + 'P',
                         'M_' + s + 'HSNUMVALID_' + y  :'M' + s + 'A', 
                         'M_' + s + 'HSPCTPROF_' + y   :'M' + s + 'P',
                         'CWD_' + s + 'HSNUMVALID_' + y:'CWD' + s + 'A', 
                         'CWD_' + s + 'HSPCTPROF_' + y :'CWD' + s + 'P',
                         'ECD_' + s + 'HSNUMVALID_' + y:'ECD' + s + 'A', 
                         'ECD_' + s + 'HSPCTPROF_' + y :'ECD' + s + 'P',
                         'LEP_' + s + 'HSNUMVALID_' + y:'LEP' + s + 'A', 
                         'LEP_' + s + 'HSPCTPROF_' + y :'LEP' + s + 'P'
                      }, inplace=True)
    
    #Select new dataframe
    salm  = salm[['STATE', 'FIPST', 
                  'LEAID', 'LEANM', 
                  'ALL' + s + 'A', 
                  'ALL' + s + 'P', 
                  'MAM' + s + 'A', 
                  'MAM' + s + 'P', 
                  'MAS' + s + 'A', 
                  'MAS' + s + 'P', 
                  'MBL' + s + 'A', 
                  'MBL' + s + 'P', 
                  'MHI' + s + 'A', 
                  'MHI' + s + 'P', 
                  'MTR' + s + 'A', 
                  'MTR' + s + 'P', 
                  'MWH' + s + 'A', 
                  'MWH' + s + 'P', 
                  'F' + s + 'A', 
                  'F' + s + 'P', 
                  'M' + s + 'A', 
                  'M' + s + 'P', 
                  'CWD' + s + 'A', 
                  'CWD' + s + 'P', 
                  'ECD' + s + 'A', 
                  'ECD' + s + 'P', 
                  'LEP' + s + 'A', 
                  'LEP' + s + 'P']]

    #Replace invalid data in the dataframe
    for col in salm.columns.values:
        salm[col] = salm[col].fillna(0)
        salm[col].replace('NaN', '')
    
    #Clean data
    salm = salm.applymap(cleandata)
    
    #Convert all column data to numeric data
    numeric = ['ALL' + s + 'P',
               'MAM' + s + 'P',
               'MAS' + s + 'P',
               'MBL' + s + 'P',
               'MHI' + s + 'P',
               'MWH' + s + 'P',
               'F' + s + 'P',
               'M' + s + 'P', 
               'CWD' + s + 'P', 
               'ECD' + s + 'P',
               'LEP' + s + 'P']
    
    #Convert each column to Numeric
    salm[numeric] = salm[numeric].apply(pd.to_numeric, errors='coerce')
    salm = salm.fillna(0)
    
    return salm[['LEAID', 'ALL' + s + 'P']]

def parseelsidata(x, y):
    elsi = pd.read_csv(x, dtype=str)

    #Add the Year column
    elsi['YEAR'] = y

    #Rename existing coulmns
    elsi.columns = ['STATE','LEAID','LEANM','TOTSC','TOTCH','TOTPS','TOTS','FRLS','GR12M','GR12F','PTR','FTE','SET','TOTSF','SEGC','YEAR']

    for col in elsi.columns.values:
            elsi[col] = elsi[col].astype('str') 
            elsi[col] = elsi[col].replace('"', '')
            elsi[col] = elsi[col].replace('†', 0)
            elsi[col] = elsi[col].replace('–', 0)
            elsi[col] = elsi[col].replace('‡', 0)
            elsi[col] = elsi[col].replace('=', '')
            elsi[col] = elsi[col].fillna(0)
            elsi[col].replace('[NaN]', '')
    
    #Convert each column to Numeric
    elsi['TOTSC'] = pd.to_numeric(elsi['TOTSC'])
    elsi['TOTCH'] = pd.to_numeric(elsi['TOTCH'])
    elsi['TOTPS'] = pd.to_numeric(elsi['TOTPS'])
    elsi['TOTS'] = pd.to_numeric(elsi['TOTS'])
    elsi['FRLS'] = pd.to_numeric(elsi['FRLS'])
    elsi['GR12M'] = pd.to_numeric(elsi['GR12M'])
    elsi['GR12F'] = pd.to_numeric(elsi['GR12F'])
    elsi['PUTR'] = pd.to_numeric(elsi['PTR'])
    elsi['FTE'] = pd.to_numeric(elsi['FTE'])
    elsi['SET'] = pd.to_numeric(elsi['SET'])
    elsi['TOTSF'] = pd.to_numeric(elsi['TOTSF'])
    elsi['SEGC'] = pd.to_numeric(elsi['SEGC'])

    #Round the data to the nearest int
    elsi['CHSR'] = round(elsi['TOTCH']/elsi['TOTSC']*100, 0)
    elsi['MLSR'] = round(elsi['GR12M']/elsi['TOTS']*100, 0)
    elsi['FLSR'] = round(elsi['GR12F']/elsi['TOTS']*100, 0)
    elsi['FRLR'] = round(elsi['FRLS']/elsi['TOTS']*100, 0)
    elsi['SETR'] = round(elsi['SET']/elsi['FTE']*100, 0)
    elsi['SECR'] = round(elsi['SEGC']/elsi['TOTSF']*100, 0)
    
    #Replace all NA values
    elsi[elsi==np.inf] = np.nan
    elsi = elsi.fillna(0)
    
    #Convert each column to integer
    elsi['CHSR'] = elsi['CHSR'].astype(int)
    elsi['MLSR'] = elsi['MLSR'].astype(int)
    elsi['FLSR'] = elsi['FLSR'].astype(int)
    elsi['FRLR'] = elsi['FRLR'].astype(int)
    elsi['SETR'] = elsi['SETR'].astype(int)
    elsi['SECR'] = elsi['SECR'].astype(int)
    
    return elsi[['LEAID','CHSR','PUTR','MLSR','FLSR','FRLR','SETR','SECR']]

acgr2011 = parsecohortdata("data/2011/acgr-lea-sy2011-12.csv", 2011)
acgr2014 = parsecohortdata("data/2014/acgr-lea-sy2014-15.csv", 2014)

admt2011 = parseassessmentdata('data/2011/math-achievement-lea-sy2011-12.csv', 2011, 'MATH')
admt2014 = parseassessmentdata('data/2014/math-achievement-lea-sy2014-15.csv', 2014, 'MATH')

adrl2011 = parseassessmentdata('data/2011/rla-achievement-lea-sy2011-12.csv', 2011, 'ENGLISH')
adrl2014 = parseassessmentdata('data/2014/rla-achievement-lea-sy2014-15.csv', 2014, 'ENGLISH')

sipe2011 = parsepovertydata("data/2011/USSD11.csv")
sipe2014 = parsepovertydata("data/2014/USSD14.csv")

elsi2011 = parseelsidata("data/2011/elsi_csv_export_2011.csv", 2011)
elsi2014 = parseelsidata("data/2014/elsi_csv_export_2014.csv", 2014)

temp1 = pd.merge(acgr2011, admt2011, how='inner', on='LEAID')
temp2 = pd.merge(temp1, adrl2011, how='inner', on='LEAID')
temp3 = pd.merge(temp2, sipe2011, how='inner', on='LEAID')
df2011 = pd.merge(temp3, elsi2011, how='inner', on='LEAID')
df2011 = df2011.rename(columns = {'ALLMTHP':'MTHP', 'ALLRLAP': 'RLAP'})

temp4 = pd.merge(acgr2014, admt2014, how='inner', on='LEAID')
temp5 = pd.merge(temp4, adrl2014, how='inner', on='LEAID')
temp6 = pd.merge(temp5, sipe2014, how='inner', on='LEAID')
df2014 = pd.merge(temp6, elsi2014, how='inner', on='LEAID')
df2014 = df2014.rename(columns = {'ALLMTHP':'MTHP', 'ALLRLAP': 'RLAP'})

print(len(df2011[df2011['ALLP'] > 100]))
print(len(df2014[df2014['ALLP'] > 100]))

print(len(df2011[df2011['ALLP'] <= 0]))
print(len(df2014[df2014['ALLP'] <= 0]))

df2011.iloc[:5,:18]

df2014.iloc[:5,:43]

null_data = df2011[df2011.isnull().any(axis=1)]
print(null_data.count)

null_data = df2014[df2014.isnull().any(axis=1)]
print(null_data.count)

df2011.to_csv('data/high-school-dropout-dataset2011.csv', mode = 'w', index=False)
df2014.to_csv('data/high-school-dropout-dataset2014.csv', mode = 'w', index=False)

