import pandas as pd

df_eb = pd.read_csv("data/out/ebola-outbreaks-before-2014-dates.csv", encoding="utf-8")

df_eb.head()

df_geom = pd.read_csv("data/in/World Country Boundaries.csv")

df_geom.head()

all_eb_countries = list(set(df_eb["Country"]))

print len(all_eb_countries)
print all_eb_countries

dict_countries = {}

for i in range(len(df_geom)):
    if df_geom["Name"][i] in all_eb_countries:
        dict_countries[df_geom["Name"][i]] = df_geom["geometry"][i]
        
print len(dict_countries)
print dict_countries.keys()

missing = set(all_eb_countries) - set(dict_countries.keys())
print len(missing)
print list(missing)

dict_codes = {
    u"C\xf4te d'Ivoire (Ivory Coast)": 'CI', 
    u'Italy': 'IT', 
    u'USA': 'US', 
    u'South Africa': 'ZA', 
    u'Democratic Republic of the Congo (formerly Zaire)': 'CD', 
    u'Philippines': 'PH', 
    u'Republic of the Congo': 'CG', 
    u'Democratic Republic of the Congo': 'CD', 
    u'Gabon': 'GA', 
    u'Sudan (South Sudan)': 'SD', 
    u'Uganda': 'UG', 
    u'Zaire (Democratic Republic of the Congo - DRC)': 'CD', 
    u'England': 'GB', 
    u'Zaire': 'CD', 
    u'Russia': 'RU'
}

## df_geom["ISO_2DIGIT"][73]

## df_geom[df_geom["ISO_2DIGIT"] == 'US']

eb_country_code = []
for country in df_eb["Country"]:
    eb_country_code.append(dict_codes.get(country))

print df_eb["Country"][:3]
print eb_country_code[:3]

df_eb.insert(7, "Country code (ISO 2 digits)", eb_country_code)

df_eb.head()

df_eb.to_csv("data/out/ebola-outbreaks-before-2014-country-codes.csv", encoding="utf-8")



