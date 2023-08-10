# Combine two dataframes into one common dataframe,
# and dump to SQL file for upload to Web app
import os
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
PROJ_ROOT = os.pardir

# For munging
import re
import json

yalePath = os.path.join(PROJ_ROOT, "data", "interim", "yale_bios_processed.csv")
harvardPath = os.path.join(PROJ_ROOT, "data", "interim", "harvard_bios_processed.p")

yale = pd.read_csv(yalePath)
harvard = pd.read_pickle(harvardPath)

yale.columns

harvard.shape

yale.columns

harvard.columns

yale["headers"][0]

harvard.head()

yale.columns

# Insert missing columns
yale["S/C"] = None
yale["College"] = "Yale"
# Re-order, then rename
# Yale
yaleColumns = ["Name", "B/T", "City", "Events", "High School", "Hometown/High School",
               "Ht.", "No.", "Pos.", "Region", "S/C", "Weapon", "Wt.", "Cl.", "season", "sport", "headers", "Bio", "College"]

yale = yale[yaleColumns]

# STILL WORKING

harvard.columns

harvard["College"] = "Harvard"
harvardColumns = ["Name", "B/T", "City", "Event", "High School", "Hometown",
               "Ht.", "No.", "Position", "Region", "S/C", "Weapon", "Wt.", "Yr.", "season", "sport", "headers", "Bio", "College"]

harvard = harvard[harvardColumns]

harvard.shape

yale.shape

# Normalize column names
normColumns = ["Name", "B/T", "City", "Events", "High School", "Hometown",
               "Ht.", "No.", "Position", "Region", "S/C", "Weapon", "Wt.", 
               "Class", "Active Seasons", "Sport", "Misc", "Bio", "College"]
yale.columns = normColumns
harvard.columns = normColumns

yale["Hometown"] = yale['Hometown'].apply(lambda x: x.split("/")[0].strip() if type(x) == str else "")

yale["Misc"][0]

stacked = pd.concat([yale, harvard], axis=0)

# Drop some of the columns we don't need
finalColumns = ["Name", "High School", "Hometown",
                "Ht.", "No.", "Position", "Wt.",
                "Active Seasons", "Misc", "Bio", "College"]
stacked = stacked[finalColumns]

stacked.head()

indices = range(0, 7274)
stacked["Student_ID"] = indices

stacked = stacked.set_index(["Student_ID"])

stacked.shape

stacked.columns

# Split up the active seasons into start and end
stacked["StartSeason"] = stacked.apply(lambda x: int(x["Active Seasons"][0:4]), axis=1)
stacked["EndSeason"] = stacked.apply(lambda x: int(x["Active Seasons"][0:2] +
                                                   x["Active Seasons"][-2:]),
                                     axis=1)

stacked = stacked.drop("Active Seasons", axis=1)

stacked.head()

# def strToDict(string):
#     if string:
#         string = string.encode('utf-8')
#         string = string.replace(" u'", " \"")
#         string = string.replace("': ", "\": ")
#         string = string.replace("', ", "\", ")
#         string = string.replace(" u\"", " \"")
#         string = string[0] + "\"" + string[3:]
#         string = string[:-2] + "\"" + string[-1]
#         try:
#             return json.loads(string, encoding="cp1252")
#         except:
#             print(string)
#             return {}
#     return {}

# stacked["Misc"] = stacked["Misc"].map(strToDict)

stacked.shape
print(json.loads(stacked["Misc"][104]))

def extractMajors(row):
    if row["Misc"] and not pd.isnull(row["Misc"]):
        misc = json.loads(row["Misc"])
        if misc and "Major:" in misc:
            return misc["Major:"]
    return ""

def height2float(height):
    if not pd.isnull(height) and height:
        pair = None
        if '-' in height:
            pair = height.split('-')
        elif "'" in height:
            pair = height.split("'")
            pair[1] = pair[1][:-1] # Remove the "
        elif "0" in height:
            pair = height.split("0")
        if len(pair) == 1:
            pair.append(float(0))
        try:
            pair = map(float, pair)              # convert strings to ints
        except:
            print(pair)
        return (12 * pair[0] + pair[1])    # assumes imperial units (12 inches per foot)  
        return -1

stacked['Ht.'] = stacked['Ht.'].map(height2float)



stacked["Major"] = stacked.apply(extractMajors, axis=1)

majors = stacked.groupby("Major").count().sort_values("Name", ascending=False)

def stripBadChars(row):
    text = row["Bio"]
    if not pd.isnull(text):
        new = re.sub("\r", "", text)
    else:
        new = ""
    return new

stacked["Bio"] = stacked.apply(stripBadChars, axis=1)

stacked.columns

# Seems about right.
processedPath = os.path.join(PROJ_ROOT, "data", "processed", "player_bios_processed.csv")
stacked.to_csv(processedPath, encoding='utf-8')

majorsPath = os.path.join(PROJ_ROOT, "data", "processed", "by_major.csv")
majors.to_csv(majorsPath, encoding='utf-8')



