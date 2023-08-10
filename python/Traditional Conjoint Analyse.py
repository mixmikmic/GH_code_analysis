import numpy as np
import pandas as pd

caInputeDF = pd.read_csv("ConjointInput.csv", sep = ";")


caInputeDF

ConjointDummyDF = pd.DataFrame(np.zeros((18,9)), columns=["Rank","A1", "A2", "A3",
                                                    "B1","B2", 
                                                    "C1", "C2",
                                                    "C3"])

ConjointDummyDF.Rank = caInputeDF.Rank

for index, row in caInputeDF.iterrows(): 
    stimuli1, stimuli2, stimuli3 = caInputeDF["Stimulus"].ix[index][:2],     caInputeDF["Stimulus"].ix[index][2:4], caInputeDF["Stimulus"].ix[index][4:6]
    
    
    ConjointDummyDF.ix[index, [stimuli1,stimuli2,stimuli3]] = 1

ConjointDummyDF.head()

fullNames = {"Rank":"Rank",            "A1": "32\" (81cm)","A2": "37\" (94cm)","A3": "42\" (107cm)",           "B1": "Plasma", "B2":"LCD",            "C1":"Silver", "C2":"Black", "C3": "Anthrazit",          }

ConjointDummyDF.rename(columns=fullNames, inplace=True)

ConjointDummyDF.head()

import statsmodels.api as sm

ConjointDummyDF.columns

X = ConjointDummyDF[[u'32" (81cm)', u'37" (94cm)', u'42" (107cm)', u'Plasma',       u'LCD', u'Silver', u'Black', u'Anthrazit']]
X = sm.add_constant(X)
Y = ConjointDummyDF.Rank
linearRegression = sm.OLS(Y, X). fit()
linearRegression.summary()

importance = []
relative_importance = []

rangePerFeature = []

begin = "A"
tempRange = []
for stimuli in fullNames.keys():
    if stimuli[0] == begin:
        tempRange.append(linearRegression.params[fullNames[stimuli]])
    elif stimuli == "Rank":
        rangePerFeature.append(tempRange)
    else:
        rangePerFeature.append(tempRange)
        begin = stimuli[0]
        tempRange = [linearRegression.params[fullNames[stimuli]]]
        

for item in rangePerFeature:
    importance.append( max(item) - min(item))

for item in importance:
    relative_importance.append(100* round(item/sum(importance),3))


partworths = []

item_levels = [1,3,5,8]

for i in range(1,4):
    part_worth_range = linearRegression.params[item_levels[i-1]:item_levels[i]]
    print part_worth_range

meanRank = []
for i in ConjointDummyDF.columns[1:]:
    newmeanRank = ConjointDummyDF["Rank"].loc[ConjointDummyDF[i] == 1].mean()
    meanRank.append(newmeanRank)

    
#total Mean or, "basic utility" is used as the "zero alternative"
totalMeanRank = sum(meanRank) / len(meanRank)



partWorths = {}
for i in range(len(meanRank)):
    name = fullNames[sorted(fullNames.keys())[i]]
    partWorths[name] = meanRank[i] - totalMeanRank

partWorths

print "Relative Importance of Feature:\n\nMonitor Size:",relative_importance[0], "%","\nType of Monitor:", relative_importance[1], "%", "\nColor of TV:", relative_importance[2], "%\n\n"

print "--"*30

print "Importance of Feature:\n\nMonitor Size:",importance[0],"\nType of Monitor:", importance[1],  "\nColor of TV:", importance[2]

#As array that looks like X
#Must include Constant!

optBundle = [1,0,0,1,0,1,0,1,0]
print "The best possible Combination of Stimuli would have the highest rank:",linearRegression.predict(optBundle)[0]

#Optimal Bundle:
#42", LCD, Black

optimalWorth = partWorths["42\" (107cm)"] + partWorths["LCD"] + partWorths["Black"]

print "Choosing the optimal Combination brings the user an additional ", optimalWorth, "'units' of utility"



