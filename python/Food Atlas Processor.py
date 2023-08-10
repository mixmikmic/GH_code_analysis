import pandas as pd
import matplotlib.pyplot as plotter
import numpy as np
import seaborn as sas
import csv
import re
sas.set()

states = {
        'AK': 'Alaska',
        'AL': 'Alabama',
        'AR': 'Arkansas',
        'AS': 'American Samoa',
        'AZ': 'Arizona',
        'CA': 'California',
        'CO': 'Colorado',
        'CT': 'Connecticut',
        'DC': 'District of Columbia',
        'DE': 'Delaware',
        'FL': 'Florida',
        'GA': 'Georgia',
        'GU': 'Guam',
        'HI': 'Hawaii',
        'IA': 'Iowa',
        'ID': 'Idaho',
        'IL': 'Illinois',
        'IN': 'Indiana',
        'KS': 'Kansas',
        'KY': 'Kentucky',
        'LA': 'Louisiana',
        'MA': 'Massachusetts',
        'MD': 'Maryland',
        'ME': 'Maine',
        'MI': 'Michigan',
        'MN': 'Minnesota',
        'MO': 'Missouri',
        'MP': 'Northern Mariana Islands',
        'MS': 'Mississippi',
        'MT': 'Montana',
        'NA': 'National',
        'NC': 'North Carolina',
        'ND': 'North Dakota',
        'NE': 'Nebraska',
        'NH': 'New Hampshire',
        'NJ': 'New Jersey',
        'NM': 'New Mexico',
        'NV': 'Nevada',
        'NY': 'New York',
        'OH': 'Ohio',
        'OK': 'Oklahoma',
        'OR': 'Oregon',
        'PA': 'Pennsylvania',
        'PR': 'Puerto Rico',
        'RI': 'Rhode Island',
        'SC': 'South Carolina',
        'SD': 'South Dakota',
        'TN': 'Tennessee',
        'TX': 'Texas',
        'UT': 'Utah',
        'VA': 'Virginia',
        'VI': 'Virgin Islands',
        'VT': 'Vermont',
        'WA': 'Washington',
        'WI': 'Wisconsin',
        'WV': 'West Virginia',
        'WY': 'Wyoming'
}

Name_Conversion = {}
with open("Data/Food_Atlas2015/Long Names.txt") as longFile:
    with open("Data/Food_Atlas2015/Short Names.txt") as shortFile:
        longReader = longFile.readlines()
        shortReader = shortFile.readlines()
        longReader = [x.strip() for x in longReader]
        shortReader = [y.strip() for y in shortReader]
        Name_Conversion = dict(zip(shortReader,longReader))
    

Name_Conversion

food_atlas_data = pd.read_csv("Data/Food_Atlas2015/Food_Atlas2015.csv")
food_atlas_data.head()

list(food_atlas_data)

food_atlas_data.rename(columns = Name_Conversion,inplace = True)

food_atlas_data.head()



code_lookups = pd.read_csv('Data/mapping_info/zip_codes_states.csv')
code_lookups.drop_duplicates(['state','county'],keep='last',inplace=True)
code_lookups['state_name'] = code_lookups.apply(lambda x: states.get(x['state'],None),axis=1)
code_lookups.head()

area_lookups = pd.read_csv('DATA/CountyAreas/DEC_10_SF1_GCTPH1.US05PR_with_ann_MOD.csv', encoding='latin-1')
code_lookups2 = code_lookups.merge(area_lookups, right_on = ['Geographic area','Geographic County'],left_on = ['state_name','county'],how='inner')
code_lookups2.head()

def plotDoubleComparison(largerFrame, numeratorValue, denominatorValue, quart=5):#"Low access, population at 1 mile for urban areas and 20 miles for rural areas, number",'Population, tract total'
    
    #if denominatorValue != 'Population, tract total' and numeratorValue!= 'Population, tract total':
    #    county_food_atlas_groupdata = largerFrame.groupby(['State','County'])[[denominatorValue,numeratorValue,'Population, tract total']].sum()
    #else:
    county_food_atlas_groupdata = largerFrame.groupby(['State','County'])[[denominatorValue,numeratorValue]].sum()
    county_food_atlas_groupdata['ratio']=county_food_atlas_groupdata[numeratorValue]/county_food_atlas_groupdata[denominatorValue]
    county_food_atlas_groupdata.reset_index(inplace=True)
    county_food_atlas_groupdata.head()
    county_food_atlas_loc_data = county_food_atlas_groupdata.merge(code_lookups, left_on = ['State','County'],right_on = ['state_name','county'],how='inner') 
    #county_food_atlas_loc_data['Population Density'] = county_food_atlas_loc_data['Population, tract total']/county_food_atlas_loc_data['Area in square miles - Land area']
    #county_food_atlas_loc_data.sort_values('Population Density',ascending = True, inplace = True)
    #if quart !=5:
    #    lowerQuart = quart-1
    #    dfSize = len(county_food_atlas_loc_data)/4
    #    county_food_atlas_loc_data = county_food_atlas_loc_data.iloc[lowerQuart*dfSize:quart*dfSize]
    county_food_atlas_loc_data.head()
    plotter.figure(figsize=(20,10))
    plotter.scatter(county_food_atlas_loc_data['longitude'], county_food_atlas_loc_data['latitude'], c=county_food_atlas_loc_data['ratio']*100.0)
    plotter.show()
    return county_food_atlas_loc_data
def plotNationalDoubleComparison(largerFrame, numeratorValue, denominatorValue):#"
    county_food_atlas_groupdata = largerFrame[[denominatorValue,numeratorValue]].sum()
    #county_food_atlas_groupdata['ratio']=county_food_atlas_groupdata[numeratorValue]/county_food_atlas_groupdata[denominatorValue]
    #county_food_atlas_groupdata.reset_index(inplace=True)
    county_food_atlas_groupdata.head()
    county_food_atlas_loc_data = county_food_atlas_groupdata#.merge(code_lookups, left_on = ['State','County'],right_on = ['state_name','county'],how='inner') 
    county_food_atlas_loc_data.head()
    #plotter.scatter(county_food_atlas_loc_data['longitude'], county_food_atlas_loc_data['latitude'], c=county_food_atlas_loc_data['ratio']*100.0)
    county_food_atlas_loc_data.plot(kind='bar')
    plotter.show()
    return county_food_atlas_loc_data
def plotNationalDoubleRatio(largerFrame, numeratorValue, denominatorValue):#"
    county_food_atlas_groupdata = largerFrame[[denominatorValue,numeratorValue]].sum()
    #county_food_atlas_groupdata['ratio']=county_food_atlas_groupdata[numeratorValue]/county_food_atlas_groupdata[denominatorValue]
    #county_food_atlas_groupdata.reset_index(inplace=True)
    county_food_atlas_groupdata.head()
    county_food_atlas_loc_data = county_food_atlas_groupdata#.merge(code_lookups, left_on = ['State','County'],right_on = ['state_name','county'],how='inner') 
    county_food_atlas_loc_data.head()
    #plotter.scatter(county_food_atlas_loc_data['longitude'], county_food_atlas_loc_data['latitude'], c=county_food_atlas_loc_data['ratio']*100.0)
    #county_food_atlas_loc_data.plot(kind='bar')
    plotter.show()
    return county_food_atlas_loc_data[numeratorValue]/county_food_atlas_loc_data[denominatorValue]

food_atlas_data[['Population, tract total','Low access, population at 1 mile for urban areas and 20 miles for rural areas, number']].head().plot()
plotter.show()

food_atlas_data2 = food_atlas_data[food_atlas_data['State']!='Alaska']
food_atlas_data2 = food_atlas_data2[food_atlas_data2['State']!='Hawaii']
food_atlas_data2 = food_atlas_data2[food_atlas_data2['State']!='Puerto Rico']


plotDoubleComparison(food_atlas_data2,'Low access, Black or African American population at 1 mile, number','Tract Black or African American population, number')

plotNationalDoubleComparison(food_atlas_data,'Low access, Black or African American population at 1 mile, number','Tract Black or African American population, number')

whiteCountyDemoBreakdown= plotDoubleComparison(food_atlas_data,'Low access, White population at 1 mile, number','Tract White population, number')
whiteCountyDemoBreakdown.head()

plotNationalDoubleComparison(food_atlas_data,'Low access, White population at 1 mile, number','Tract White population, number')

plotDoubleComparison(food_atlas_data,'Low access, Hispanic or Latino population at 1 mile, number', 'Tract Hispanic or Latino population, number')
plotNationalDoubleComparison(food_atlas_data,'Low access, Hispanic or Latino population at 1 mile, number', 'Tract Hispanic or Latino population, number')

plotDoubleComparison(food_atlas_data2,'Low access, population at 1 mile, number','Population, tract total')

whiteCountyDemoBreakdown['testRatio'] = whiteCountyDemoBreakdown.apply(lambda x: 0.1 if x['longitude']<-100. else 0.9, axis = 1)
plotter.figure(figsize=(20,10))
plotter.scatter(whiteCountyDemoBreakdown['longitude'], whiteCountyDemoBreakdown['latitude'], c=whiteCountyDemoBreakdown['testRatio']*100.0)
plotter.show()

#racial breakdown over 1 mile from a grocer
pd.DataFrame(pd.Series({'Hispanic': 100* plotNationalDoubleRatio(food_atlas_data,'Low access, Hispanic or Latino population at 1 mile, number', 'Tract Hispanic or Latino population, number'),
'White': 100*plotNationalDoubleRatio(food_atlas_data,'Low access, White population at 1 mile, number','Tract White population, number'),
 'Black' : 100 * plotNationalDoubleRatio(food_atlas_data,'Low access, Black or African American population at 1 mile, number','Tract Black or African American population, number')})).plot(kind = 'bar')
plotter.show()

def racialQuartilePercents(quart=1):
    county_food_atlas_groupdata2 = food_atlas_data.groupby(['State','County'])[['Low access, Hispanic or Latino population at 1 mile, number', 'Tract Hispanic or Latino population, number','Low access, White population at 1 mile, number','Tract White population, number','Low access, Black or African American population at 1 mile, number','Tract Black or African American population, number','Population, tract total']].sum()
    #county_food_atlas_groupdata2['Hispanic_Ratio']=county_food_atlas_groupdata2['Low access, Hispanic or Latino population at 1 mile, number']/county_food_atlas_groupdata2['Tract Hispanic or Latino population, number']
    #county_food_atlas_groupdata2['White_Ratio']=county_food_atlas_groupdata2['Low access, White population at 1 mile, number']/county_food_atlas_groupdata2['Tract White population, number']
    #county_food_atlas_groupdata2['African_American_Ratio']=county_food_atlas_groupdata2['Low access, Black or African American population at 1 mile, number']/county_food_atlas_groupdata2['Tract Black or African American population, number']
    county_food_atlas_groupdata2.reset_index(inplace=True)
    county_food_atlas_groupdata2.head()
    county_food_atlas_loc_data2 = county_food_atlas_groupdata2.merge(code_lookups2, left_on = ['State','County'],right_on = ['state_name','county'],how='inner') 
    county_food_atlas_loc_data2['Population Density'] = county_food_atlas_loc_data2['Population, tract total']/county_food_atlas_loc_data2['Area in square miles - Land area']
    county_food_atlas_loc_data2.sort_values('Population Density',ascending = True, inplace = True)
    dfSize =int( len(county_food_atlas_loc_data2)/4)
    #quart=2
    if quart !=5:
        lowerQuart = quart-1
        county_food_atlas_loc_data2 = county_food_atlas_loc_data2.iloc[lowerQuart*dfSize:quart*dfSize]
    county_food_atlas_loc_data2.head()
    plotter.figure(figsize=(10,5))
    #plotter.scatter(county_food_atlas_loc_data2['longitude'], county_food_atlas_loc_data2['latitude'], c=county_food_atlas_loc_data2['White_Ratio']*100.0)
    new_bars = county_food_atlas_loc_data2[['Low access, Hispanic or Latino population at 1 mile, number', 'Tract Hispanic or Latino population, number','Low access, White population at 1 mile, number','Tract White population, number','Low access, Black or African American population at 1 mile, number','Tract Black or African American population, number']]#'Hispanic_Ratio','White_Ratio','African_American_Ratio']]

    new_bars=new_bars.sum()
    new_bars['Hispanic_Ratio']=100*new_bars['Low access, Hispanic or Latino population at 1 mile, number']/new_bars['Tract Hispanic or Latino population, number']
    new_bars['White_Ratio']=100*new_bars['Low access, White population at 1 mile, number']/new_bars['Tract White population, number']
    new_bars['African_American_Ratio']=100*new_bars['Low access, Black or African American population at 1 mile, number']/new_bars['Tract Black or African American population, number']

    new_bars = new_bars[['Hispanic_Ratio','White_Ratio','African_American_Ratio']]

    new_bars.head()
    return new_bars
    

plotter.bar(['Hispanic','White non-Hispanic','Black non-Hispanic'],racialQuartilePercents(1).values.tolist(),color=['slategray','goldenrod','teal'])
plotter.title('Population Percent Over One Mile From Access, Lowest Quartile Pop. Density')
plotter.ylim(0,70)
plotter.yticks(fontSize=16)
plotter.xticks(fontSize=20)
#ax.set_xticklabels(['Hispanic','White non-Hispanic','Black non-Hispanic'],rotation='horizontal')#'Hispanic_Ratio','White_Ratio','African_American_Ratio'
plotter.show()

plotter.bar(['Hispanic','White non-Hispanic','Black non-Hispanic'],racialQuartilePercents(2).values.tolist(),color=['slategray','goldenrod','teal'])
plotter.title('Population Percent Over One Mile From Access')
plotter.ylim(0,70)
plotter.yticks(fontSize=16)
plotter.xticks(fontSize=20)
#ax.set_xticklabels(['Hispanic','White non-Hispanic','Black non-Hispanic'],rotation='horizontal')#'Hispanic_Ratio','White_Ratio','African_American_Ratio'
plotter.show()

plotter.bar(['Hispanic','White non-Hispanic','Black non-Hispanic'],racialQuartilePercents(3).values.tolist(),color=['slategray','goldenrod','teal'])
plotter.title('Population Percent Over One Mile From Access')
plotter.ylim(0,70)
plotter.yticks(fontSize=16)
plotter.xticks(fontSize=20)
#ax.set_xticklabels(['Hispanic','White non-Hispanic','Black non-Hispanic'],rotation='horizontal')#'Hispanic_Ratio','White_Ratio','African_American_Ratio'
plotter.show()

plotter.bar(['Hispanic','White non-Hispanic','Black non-Hispanic'],racialQuartilePercents(4).values.tolist(),color=['slategray','goldenrod','teal'])
plotter.title('Population Percent Over One Mile From Access, Highest Quartile Pop. Density')
plotter.ylim(0,70)
plotter.yticks(fontSize=16)
plotter.xticks(fontSize=20)
#ax.set_xticklabels(['Hispanic','White non-Hispanic','Black non-Hispanic'],rotation='horizontal')#'Hispanic_Ratio','White_Ratio','African_American_Ratio'
plotter.show()

#fig,ax = plotter.subplots()
#racialQuartilePercents(5).plot(kind='bar', title='Population Percent Over One Mile From Access')
plotter.bar(['Hispanic','White non-Hispanic','Black non-Hispanic'],racialQuartilePercents(5).values.tolist(),color=['slategray','goldenrod','teal'])
plotter.title('Population Percent Over One Mile From Access',fontSize=24)
plotter.ylim(0,70)
plotter.yticks(fontSize=16)
plotter.xticks(fontSize=20)
#ax.set_xticklabels(['Hispanic','White non-Hispanic','Black non-Hispanic'],rotation='horizontal')#'Hispanic_Ratio','White_Ratio','African_American_Ratio'
plotter.show()

np.transpose(racialQuartilePercents(5).values.tolist())

racialQuartilePercents(5).plot(kind='bar')
plotter.show()



chsi_data = pd.read_csv("DATA/communityHealthStatusIndicators_dataset/RISKFACTORSANDACCESSTOCARE.csv")
chsi_data.head()


tempGroup = food_atlas_data.groupby(['State','County'])[['Low access, population at 1 mile, number','Population, tract total']].sum()
tempGroup['ratio']=tempGroup['Low access, population at 1 mile, number']/tempGroup['Population, tract total']
tempGroup.reset_index(inplace=True)
tempGroup.head()
temp_food_loc = tempGroup.merge(code_lookups, left_on = ['State','County'],right_on = ['state_name','county'],how='inner') 
temp_food_loc_health = temp_food_loc.merge(chsi_data,left_on = ['state_name','county'], right_on = ['CHSI_State_Name','CHSI_County_Name'],how='inner')
temp_food_loc_health.head()
#county_food_atlas_loc_data['Population Density'] = county_food_atlas_loc_data['Population, tract total']/county_food_atlas_loc_data['Area in square miles - Land area']
#county_food_atlas_loc_data.sort_values('Population Density',ascending = True, inplace = True)
#if quart !=5:
#    lowerQuart = quart-1
#    dfSize = len(county_food_atlas_loc_data)/4
#    county_food_atlas_loc_data = county_food_atlas_loc_data.iloc[lowerQuart*dfSize:quart*dfSize]
#county_food_atlas_loc_data.head()
#temp_food_loc_health['']
plotter.figure(figsize=(20,10))
temp_food_loc_health=temp_food_loc_health[temp_food_loc_health['Obesity']>=0]
plotter.scatter(temp_food_loc_health['ratio']*100, temp_food_loc_health['Obesity'])
plotter.xlabel("Scarcity Ratio",fontSize=24)
plotter.ylabel('Obesity Rate',fontSize=24)
plotter.title('Obesity Rate x Scarcity By County',fontSize=24)
plotter.show()
#return county_food_atlas_loc_data

tempGroup['Low access tract at 1 mile'].sum()



