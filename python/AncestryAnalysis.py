import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from chorogrid import Colorbin, Chorogrid  # For plotting choropleth maps
from colour import Color  # Easily creating colors for the maps

plt.style.use('seaborn-deep')  # Keeps plot aesthetics consistent
pd.set_option('display.max_columns', None)  # Displays all columns for wide data frames

get_ipython().magic('matplotlib inline')

# Individual Detail
df = pd.read_csv('Macaluso_Tree.csv', nrows=489)
df.dropna(axis = 'columns', how='all', inplace=True)  # Drops any column that doesn't contain any data
df.info()

# Marriage Detail
Marriage = pd.read_csv('Macaluso_Tree.csv', skiprows=490, nrows=87)
Marriage.dropna(axis='columns', how='all', inplace=True)

# Family Detail
Family = pd.read_csv('Macaluso_Tree.csv', skiprows=579, nrows=407)
Family.dropna(axis='columns', how='all', inplace=True)

# Too few sources to bother with
df.drop(['Birth source', 'Death source', 'Burial source'], axis=1, inplace=True)

# Create a new column for full name.  Correcting often misspelled last name.
df.replace(to_replace='MacAluso', value='Macaluso', inplace=True)  # No, it's not Scottish or Irish
df["Name"] = df["Given"].map(str) + " " + df["Surname"]

# Renaming the columns to make them more intuitive
Family.columns = ['Family', 'Person']

Father_Info = df.drop(['Surname', 'Given', 'Suffix', 'Gender'], axis=1)

# Columns will be renamed in the final join
Father_Info.columns = ['Husband', 'Father Birth date', 'Father Birth place',
                       'Father Death date', 'Father Death place',
                       'Father Burial place', 'Father Name']

Father = pd.concat([Marriage['Husband']], axis=1)  # Inner join to limit the list to only husbands

Father = Father.merge(Father_Info, on='Husband')

Father.tail()

Mother_Info = df.drop(['Surname', 'Given', 'Suffix', 'Gender'], axis=1)

# Columns will be renamed in the final join
Mother_Info.columns = ['Wife', 'Mother Birth date', 'Mother Birth place', 'Mother Death date',
                       'Mother Death place', 'Mother Burial place', 'Mother Name']

Mother = pd.concat([Marriage['Wife']], axis=1)

Mother = Mother.merge(Mother_Info, on='Wife')

Mother.tail()

# Merge the Family data frame to assign the family ID for other joins
df = pd.merge(df, Family, on='Person', how='left')

# Merge the Marriage data frame to assign the mother and father
Marriage.columns = ['Family', 'FatherKey', 'MotherKey', 'Date', 'Place', 'Source']
df = pd.merge(df, Marriage, on='Family', how='left')

# Merge the parental info on the data frame
df = pd.merge(df, Father, how ='left', left_on='FatherKey', right_on='Husband')
df = pd.merge(df, Mother, how ='left', left_on='MotherKey', right_on='Wife')
df.drop(['Husband', 'Wife', 'MotherKey', 'FatherKey', 'Suffix', 'Date', 'Place', 'Source'], 
        axis=1, inplace=True)
df.drop_duplicates(inplace=True)  # To account for dupliciate rows later discovered
df.reset_index(drop=True, inplace=True)

import missingno as msno

msno.matrix(df)

# Renaming columns to remove spaces for dot notation
df.columns = ['PersonKey', 'Surname', 'Given', 'Gender', 'BirthDate', 'BirthPlace',
              'DeathDate', 'DeathPlace', 'BurialPlace', 'Name', 'Family',
              'FatherBirthDate', 'FatherBirthPlace', 'FatherDeathDate','FatherDeathPlace',
              'FatherBurialPlace', 'FatherName', 'MotherBirthDate', 'MotherBirthPlace',
              'MotherDeathDate', 'MotherDeathPlace',
              'MotherBurialPlace', 'MotherName']

df.ix[304:307]

# Splitting to account for middle names listed within the 'Given' column
family_names = [str(name).split() for name in df['Given'].dropna()]

# Flattening, removing abbreviations, and transforming into a series to speed things up
family_names = [item for sublist in family_names for item in sublist]
family_names = pd.Series([word if len(word) > 2 else np.NaN for word in family_names]).dropna()

family_names = family_names.value_counts()

# Top 10 names
family_names[:10]

family_names[family_names.index == 'Bryan']

ax = family_names[:20].plot(kind="barh", figsize=(10, 7), title="Family Names")
ax.set_xlabel("Occurrences")

plt.gca().invert_yaxis()  # For descending order

df[df['Given'].str.contains("Keturah") == True]

# Frequency of surnames
Surname = df[['Surname', 'PersonKey']].dropna()
Surname = Surname.groupby('Surname').count()
Surname = Surname.sort_values(by='PersonKey', ascending=0)
ax = Surname[:15].plot(kind='barh', figsize=(10, 7), legend=False, title='Last Names')
ax.set_xlabel("Occurrences")
ax.invert_yaxis()

# Creating a dictionary of states and their codes to extract places into states
states = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missisippi': 'MS',  # to account for spelling errors
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY',
}

# Reverses the above dictionary to extract state abbreviations using the same function
states_reverse = dict(zip(states.values(),states.keys()))

# Extracting the state from various addresses
def state_extract(dict1, dict2, column):
    """
    Extracts the state from the place columns if any part of the record exists 
    within the state dictionary or reverse state dictionary
    
    dict1 is the regular state dictionary, dict2 is the reverse state dictionary for state abbreviations
    """
    return pd.Series([[state.strip() for state in cell.split(',')
               if state.strip() in dict1
                   or state.strip().upper() in dict2]  # dict2 for state abbreviations
              for cell in column.fillna("null")])
    
    
# Running functions to extract the states into series
BirthState = state_extract(states, states_reverse, df['BirthPlace'])
DeathState = state_extract(states, states_reverse, df['DeathPlace'])

FatherBirthState = state_extract(states, states_reverse, df['FatherBirthPlace'])
FatherDeathState = state_extract(states, states_reverse, df['FatherDeathPlace'])

MotherBirthState = state_extract(states, states_reverse, df['MotherBirthPlace'])
MotherDeathState = state_extract(states, states_reverse, df['MotherDeathPlace'])


# Assigns series generated from the function to columns and removes brackets surrounding values
df['BirthState'] = BirthState.apply(lambda s: s[-1] if s else np.NaN)
df['DeathState'] = DeathState.apply(lambda s: s[-1] if s else np.NaN)

df['FatherBirthState'] = FatherBirthState.apply(lambda s: s[-1] if s else np.NaN)
df['FatherDeathState'] = FatherDeathState.apply(lambda s: s[-1] if s else np.NaN)

df['MotherBirthState'] = MotherBirthState.apply(lambda s: s[-1] if s else np.NaN)
df['MotherDeathState'] = MotherDeathState.apply(lambda s: s[-1] if s else np.NaN)


# Converts from abbreviated state to full state name in title case
df['BirthState'] = df['BirthState'].str.upper().replace(states_reverse).str.title()
df['DeathState'] = df['DeathState'].str.upper().replace(states_reverse).str.title()

df['FatherBirthState'] = df['FatherBirthState'].str.upper().replace(states_reverse).str.title()
df['FatherDeathState'] = df['FatherDeathState'].str.upper().replace(states_reverse).str.title()

df['MotherBirthState'] = df['MotherBirthState'].str.upper().replace(states_reverse).str.title()
df['MotherDeathState'] = df['MotherDeathState'].str.upper().replace(states_reverse).str.title()

df[['BirthPlace', 'BirthState', 'DeathPlace', 'DeathState', 'FatherBirthState', 'MotherBirthState']].ix[:3]

# Adjust for "Missisippi" spelling error
df.replace("Missisippi", "Mississippi", inplace=True)

# Adding abbreviated columns for choropleth maps later
df['BirthStateAbbrev'] = df['BirthState'].replace(states)
df['DeathStateAbbrev'] = df['DeathState'].replace(states)

fig, axs = plt.subplots(ncols=2, sharey=True)
sns.countplot(y='BirthState', data=df, ax=axs[0])
sns.countplot(y='DeathState', data=df, ax=axs[1])
fig.set_size_inches(8,8)

# Creates binary columns to show if the individual died where they were born and if they're currently alive
df['BirthState'].fillna(np.NaN)
df['DiedWhereBorn'] = np.where(df['BirthState'] == df['DeathState'], 1,
                               np.where(df['DeathState'].isnull(), np.NaN, 0))  # Also add NaN
df['IsAlive'] = np.where(df['DeathDate'].isnull(), 1, 0)

# Creates binary columns for if the individual was born in their mother or father's states.  
# Will be used to calculate mobility
df['BornInFatherState'] = np.where(df['BirthState'] == df['FatherBirthState'], 1,
                                   np.where(df['FatherBirthState'].isnull(), np.NaN, 0))
df['BornInMotherState'] = np.where(df['BirthState'] == df['MotherBirthState'], 1,
                                   np.where(df['MotherBirthState'].isnull(), np.NaN, 0))

fig, ax = plt.subplots()

sns.countplot(x='DiedWhereBorn', data=df)

countries = {
    # Europe
    'England': 'EN',
    'Scotland': 'SC',
    'Wales': 'WL',
    'Ireland': 'IR',
    'Rebpulic of Ireland': 'IR',
    'Great Britain': 'GB',
    'Britain': 'GB',
    'United Kingdom': 'UK',
    'Italy': 'IT',
    'ITA': 'IT',
    'Germany': 'DE',
    'France': 'FR',
    'Denmark': 'DR',
    'Sweden': 'SW',
    'Norway': 'NR',
    'Canada': 'CN',
    'Netherlands': 'ND',
    'Spain': 'SP',
    'Belgium': 'BG',
    'Poland': 'PL',
    'Austria': 'AR',
    'Portugal': 'PR',
    'Russia': 'RU',
    'Hungary': 'HR',
    'Slovakia': 'SL',
    'Greece': 'GR',
    
    # North America
    'United States': 'US',
    'USA': 'US',
    'US': 'US',
    'Mexico': 'MX'
}

# Extracting the country from birth addresses
BirthCountry = pd.Series([[state.strip() for state in cell.split(',') 
                  if state.strip() in countries] 
                 for cell in df.BirthPlace.fillna("null")])

df['BirthCountry'] = BirthCountry.apply(lambda s: s[-1] if s else np.NaN)
df['BirthCountry'] = np.where(df['BirthCountry'] == 'USA', 'United States', df['BirthCountry'])

fig, ax = plt.subplots()

fig.set_size_inches(10, 5)
sns.countplot(y='BirthCountry', data=df)

DiedWhereBornRatio = df['DiedWhereBorn'].groupby(df['BirthState']).mean()
DiedWhereBornRatio.plot(kind="barh", figsize=(10, 6), 
                        title="Died in Same State as Birth").set_xlabel("Ratio")

FatherStateBirthRatio = df['BornInFatherState'].groupby(df['BirthState']).mean()
MotherStateBirthRatio = df['BornInMotherState'].groupby(df['BirthState']).mean()

BirthRatioByState = pd.concat([FatherStateBirthRatio, MotherStateBirthRatio], axis=1)
BirthRatioByState.columns = ['Father', 'Mother']

ax = BirthRatioByState.plot(kind="barh", figsize=(10, 6),
                            title="Born in Same State as Parents")
ax.set_xlabel("Ratio")
ax.set_ylabel("State")

# Binning the birth years into decades

Years_int = np.arange(1500, 2020)  # To extract the birth and death years
Years = [str(s) for s in Years_int]  # Label argument to pass

decades = np.arange(1600, 2020, 10)
decades_labels = [s for s in decades]
del decades_labels[-1]  # Removes the last record to use as the labels in the pd.cut


# Extracting the birth year from the date
BirthYear = pd.Series([[date.strip() for date in cell.split(' ')
                  if date.strip() in Years]
                 for cell in df['BirthDate'].fillna("null").str.replace('-', ' ')])

# Applying the bins
df['BirthYear'] = BirthYear.apply(lambda s: s[-1] if s else np.NaN)
df['BirthYear'] = pd.to_numeric(df['BirthYear'])
df['BirthDecade'] = pd.cut(df['BirthYear'], decades, labels=decades_labels).astype(float)

# Extracting the death year from the date
DeathYear = pd.Series([[date.strip() for date in cell.split(' ')
                  if date.strip() in Years]
                 for cell in df['DeathDate'].fillna("null").str.replace('-', ' ')])

# Appending to the data frame as a numeric column
df['DeathYear'] = DeathYear.apply(lambda s: s[-1] if s else np.NaN)
df['DeathYear'] = pd.to_numeric(df['DeathYear'])

df[['BirthDate', 'BirthYear', 'BirthDecade', 'DeathDate', 'DeathYear']].head()

died_where_born_ratio_ts = df['DiedWhereBorn'].groupby(df['BirthDecade']).mean().dropna()

died_where_born_ratio = died_where_born_ratio_ts.reset_index()
died_where_born_ratio.columns = ['BirthDecade', 'Ratio']
died_where_born_ratio.dropna(inplace = True)


ax = died_where_born_ratio_ts.plot(figsize=(10, 5))
# Plotting a LOWESS line on top with seaborn's regplot() command
sns.regplot(x='BirthDecade', y='Ratio', data=died_where_born_ratio, lowess=True, color='Green')

ax.set_ylim([-0.01, 1.01])
ax.set_title('Mobility: Died in the Same State as Birth by Decade')
ax.set_xlabel("Decade Born")

born_in_father_state_ratio = df['BornInFatherState'].groupby(df['BirthDecade']).mean()
born_in_mother_state_ratio = df['BornInMotherState'].groupby(df['BirthDecade']).mean()
BirthRatio = pd.concat([born_in_father_state_ratio, born_in_mother_state_ratio], axis=1)
BirthRatio.columns = ['Father', 'Mother']

# Split our data between mother and father to create our flags
birth_ratio_linear = BirthRatio
linear_father = birth_ratio_linear[['Father']]
linear_father.columns = ['Ratio']
linear_father['Parent'] = 'Father'

linear_mother = birth_ratio_linear[['Mother']]
linear_mother.columns = ['Ratio']
linear_mother['Parent'] = 'Mother'

# Recombine, drop NA, reset index to convert it to a data frame
birth_ratio_linear = linear_father.append(linear_mother)
birth_ratio_linear.dropna(inplace=True)
birth_ratio_linear.reset_index(inplace=True)


ax = BirthRatio.dropna().plot(title="Children Born in the Same State as Parents", figsize=(10, 5))

sns.regplot(x="BirthDecade", y="Ratio",
                data=birth_ratio_linear[birth_ratio_linear['Parent'] == 'Mother'],
               lowess=True, color='LightGreen')

sns.regplot(x="BirthDecade", y="Ratio",
                data=birth_ratio_linear[birth_ratio_linear['Parent'] == 'Father'],
               lowess=True, color='LightBlue')

ax.set_ylabel("Ratio")
ax.set_xlabel("Decade")

# Using static year instead of datetime.date.today().year since data is static
df['LifeSpan'] = df['DeathYear'] - df['BirthYear']

df['CurrentAge'] = (df[(df['DeathYear'].isnull()) & 
                      (df['BirthYear'] >= (2016-100))].DeathYear.fillna(2016) 
                   - 
                   df[(df['DeathYear'].isnull()) & 
                      (df['BirthYear'] >= (2016-115))].BirthYear)

lifespan = df[df['LifeSpan'].isnull() == False][['LifeSpan', 'BirthDecade']]
avg_lifespan_decade = lifespan['LifeSpan'].groupby(lifespan['BirthDecade']).mean()
avg_lifespan_decade = avg_lifespan_decade.dropna().iloc[:-1]  # Removing the last record to prevent skew

ax = avg_lifespan_decade.plot(figsize=(10, 5))
# Plotting a LOWESS line on top with seaborn's regplot() command
sns.regplot(x='BirthDecade', y='LifeSpan', data=avg_lifespan_decade.reset_index(),
            lowess=True, color='Green')
ax.set_ylim([0,80])
ax.set_title('Life Span by Decade')
ax.set_ylabel("Life Span")
ax.set_xlabel("Decade Born")

def hex_to_RGB(hex):
  """ 
  "#FFFFFF" -> [255,255,255] 
  """
  # Pass 16 to the integer function for change of base
  return [int(hex[i:i+2], 16) for i in range(1,6,2)]

def RGB_to_hex(RGB):
  """ 
  [255,255,255] -> "#FFFFFF" 
  """
  # Components need to be integers for hex to make sense
  RGB = [int(x) for x in RGB]
  return "#"+"".join(["0{0:x}".format(v) if v < 16 else
            "{0:x}".format(v) for v in RGB])

def color_dict(gradient):
  """ 
  Takes in a list of RGB sub-lists and returns dictionary of
  colors in RGB and hex form for use in a graphing function
  defined later on 
  """
  return {"hex":[RGB_to_hex(RGB) for RGB in gradient],
      "r":[RGB[0] for RGB in gradient],
      "g":[RGB[1] for RGB in gradient],
      "b":[RGB[2] for RGB in gradient]}

def linear_gradient(start_hex, finish_hex="#FFFFFF", n=10):
  """ 
  Returns a gradient list of (n) colors between
  two hex colors. start_hex and finish_hex
  should be the full six-digit color string,
  inlcuding the number sign ("#FFFFFF") 
  """
  # Starting and ending colors in RGB form
  s = hex_to_RGB(start_hex)
  f = hex_to_RGB(finish_hex)
  # Initilize a list of the output colors with the starting color
  RGB_list = [s]
  # Calcuate a color at each evenly spaced value of t from 1 to n
  for t in range(1, n):
    # Interpolate RGB vector for color at the current value of t
    curr_vector = [
      int(s[j] + (float(t)/(n-1))*(f[j]-s[j]))
      for j in range(3)
    ]
    # Add it to our list of output colors
    RGB_list.append(curr_vector)

  return color_dict(RGB_list)


mapGreen = linear_gradient('#bfe0be', '#2a6828', 6)['hex']
mapRed = linear_gradient('#e59292', '#5b0505', 6)['hex']
mapBlue = linear_gradient('#3ca9f2', '#2c3e50', 6)['hex']
mapPurple = linear_gradient('#c295d8', '#5f018e', 6)['hex']

def mapLegendScale(column):
    """
    Returns a list of values to discretize the legend in the choropleth maps
    """
    maxVal = column.max()
    legend = [int(x) for x in list(np.arange(0, maxVal+1, maxVal/6))]
    return legend

def mapLegendScaleMinStart(column):
    """
    Returns a list of values to discretize the legend in the choropleth maps
    
    Starts with the minimum rather than 0
    """
    maxVal = column.max()
    minVal = column.min()
    legend = [int(x) for x in list(np.arange(minVal, maxVal+1, (maxVal-minVal)/6))]
    return legend

def createMapUSA(column, colors, title, legendTitle, legendScaleMin=0):
    """
    Creates and plots a choropleth map for the USA based off of the given inputs
    
    Must come from the usaMap data frame
    """
    # Creating the color bin per the chorogrid documentation
    mapBin = Colorbin(usaMap[column].dropna(), colors, proportional=True, decimals=None)
    
    # Creates the intervals for the legend using the previously defined functions
    # Will use the appropriate function given the input
    if legendScaleMin == 1:
        mapBin.fenceposts = mapLegendScaleMinStart(usaMap[column].dropna())
    else:
        mapBin.fenceposts = mapLegendScale(usaMap[column].dropna())
        if legendScaleMin != 0:
            print('Warning: Expecting binary input.  Assuming 0.')
    mapBin.recalc(False)
    
    # Applies colors to the states
    statesMap = list(usaMap.dropna(subset=[column]).State)
    colors_by_state = mapBin.colors_out
    font_colors_by_state = mapBin.complements
    legend_colors = mapBin.colors_in
    legend_labels = mapBin.labels
    
    # Creates the actual plot
    cg = Chorogrid('usa_states.csv', statesMap, colors_by_state)
    cg.set_title(title)
    cg.set_legend(mapBin.colors_in, mapBin.labels, title=legendTitle)
    cg.draw_map(spacing_dict={'legend_offset': [-275, -200]})
    cg.done(show=True)

# Creating one data frame for the USA maps from different groupbys
birthStates = df['PersonKey'].groupby(df['BirthStateAbbrev']).count()
birthStates = birthStates.reset_index()
birthStates.columns = ['State', 'BirthCount']  # Renaming for the merge

deathStates = df['PersonKey'].groupby(df['DeathStateAbbrev']).count()
deathStates = deathStates.reset_index()
deathStates.columns = ['State', 'DeathCount']  # Renaming for the merge

timeStates = df['BirthYear'].groupby(df['BirthStateAbbrev']).mean()
timeStates = timeStates.reset_index()
timeStates.columns = ['State', 'AvgYearBorn']  # Renaming for the merge

lifeSpanStates = df['LifeSpan'].groupby(df['BirthStateAbbrev']).mean().dropna()
lifeSpanStates = lifeSpanStates.reset_index()
lifeSpanStates.columns = ['State', 'AvgLifeSpan']  # Renaming for the merge


# Gathering a percentage of residents from the state that moved
staticStates = df[df['DiedWhereBorn'] == 1]['PersonKey'].groupby(df['BirthStateAbbrev']).count()     / df['PersonKey'].groupby(df['BirthStateAbbrev']).count() * 100
staticStates = staticStates.reset_index().dropna()
staticStates.columns = ['State', 'DiedWhereBornPct']  # Renaming for the merge

emigratedFromStates = df[df['DiedWhereBorn'] == 0]['PersonKey'].groupby(df['BirthStateAbbrev']).count()     / df['PersonKey'].groupby(df['BirthStateAbbrev']).count() * 100
emigratedFromStates = emigratedFromStates.reset_index().dropna()
emigratedFromStates.columns = ['State', 'EmigratedPct']

immigratedToStates = df[df['DiedWhereBorn'] == 0]['PersonKey'].groupby(df['DeathStateAbbrev']).count()
immigratedToStates = immigratedToStates.reset_index().dropna()
immigratedToStates.columns = ['State', 'ImmigratedCount']


usaMap = pd.merge(birthStates, deathStates, how='outer')
usaMap = usaMap.merge(staticStates, how='outer').merge(timeStates, how='outer')                .merge(lifeSpanStates, how='outer').merge(emigratedFromStates, how='outer')                .merge(immigratedToStates, how='outer')

# Applying percentages to normalize our maps
usaMap['BirthPct'] = usaMap['BirthCount']/usaMap['BirthCount'].sum() * 100
usaMap['DeathPct'] = usaMap['DeathCount']/usaMap['DeathCount'].sum() * 100
usaMap['ImmigratedPct'] = usaMap['ImmigratedCount']/usaMap['ImmigratedCount'].sum() * 100

usaMap.head()

createMapUSA('BirthPct',  # Column to plot
             mapGreen,  # Colors to use
             'Birth',  # Title
             'Ancestors Born %',  # Legend Title
             legendScaleMin=0)  # Binary for if the lowest value should be the min value or 0

createMapUSA('AvgLifeSpan',  # Column to plot
             mapPurple,  # Colors to use
             'Average Life Span',  # Title
             'Life Span (Yrs.)',  # Legend Title
             legendScaleMin=1)  # Binary for if the lowest value should be the min value or 0

createMapUSA('AvgYearBorn',  # Column to plot
             mapBlue[::-1],  # Colors to use
             'Migratory Patterns',  # Title
             'Avg Birth Year',  # Legend Title
             legendScaleMin=1)  # Binary for if the lowest value should be the min value or 0

createMapUSA('DiedWhereBornPct',  # Column to plot
             mapPurple,  # Colors to use
             'Never Moved Away',  # Title
             'Died Where Born %',  # Legend Title
             legendScaleMin=0)  # Binary for if the lowest value should be the min value or 0

createMapUSA('EmigratedPct',  # Column to plot
             mapRed,  # Colors to use
             'Least Desirable States',  # Title
             'Emigrated From %',  # Legend Title
             legendScaleMin=0)  # Binary for if the lowest value should be the min value or 0

createMapUSA('ImmigratedPct',  # Column to plot
             mapGreen,  # Colors to use
             'Most Desirable States',  # Title
             'Immigrated To %',  # Legend Title
             legendScaleMin=0)  # Binary for if the lowest value should be the min value or 0

createMapUSA('DeathPct',  # Column to plot
             mapRed,  # Colors to use
             'Death',  # Title
             'Ancestors Died %',  # Legend Title
             legendScaleMin=0)  # Binary for if the lowest value should be the min value or 0

euroMap = df['BirthCountry'].groupby(df['BirthCountry']).count()
euroMap.name = 'BirthCount'
euroMap = euroMap.reset_index()

# Removing non-European countries
euroMap = euroMap[(euroMap['BirthCountry'] != 'United States') & 
                    (euroMap['BirthCountry'] != 'Canada')]

def countryAbbrev(country):
    """
    Manually converting country names according to chorogrid's European country 
    name abbreviations due to dictionary containing multiple abbreviations per country
    """
    if country == 'England' or country == 'Scotland' or country == 'Wales':
        return 'GB'
    if country == 'France':
        return 'FR'
    if country == 'Italy':
        return 'IT'
    else:
        return np.NaN
    
euroMap['Country'] = euroMap['BirthCountry'].apply(lambda country: countryAbbrev(country))

euroMap

# Re-aggregating after England/Scotland/Wales were assigned GB
euroMap = euroMap.groupby('Country').sum().reset_index()

euroMap['AncestorPct'] = euroMap['BirthCount']/euroMap['BirthCount'].sum() * 100

# Adjusting for under-represented Italian ancestry
nonITCount = euroMap[euroMap['Country'] != 'IT']['BirthCount'].sum()
euroMap['AdjustedPct'] = np.where(euroMap['Country'] == 'IT', 50,
                                  (euroMap['BirthCount']/nonITCount * 100)/2)

euroMap

mapBin = Colorbin(euroMap['AdjustedPct'], mapGreen, proportional=False, decimals=None)
mapBin.fenceposts = mapLegendScale(euroMap['AncestorPct'])
mapBin.recalc(False)

cg = Chorogrid('europe_countries.csv', euroMap.Country, mapBin.colors_out, 'abbrev')
cg.set_title('Ancestral Countries')
cg.set_legend(mapBin.colors_in, mapBin.labels, title='% Ancestry')
cg.draw_map(spacing_dict={'legend_offset':[-200,-100], 'margin_top': 50})
cg.done(show=True)

