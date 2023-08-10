import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

get_ipython().magic('matplotlib inline')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

get_ipython().system('ls')

df = pd.read_csv("country-gdp-2014.csv")
df.head()

df.groupby("Continent")['life_expectancy'].median().plot(kind = 'barh')

ax = df.groupby("Continent")['life_expectancy'].median().plot(kind = 'barh')
ax.set_ylabel("")

# Initialize the plot first, take the ax
fig, ax = plt.subplots(figsize=(10,5))

# Pass the ax to the  .plot function
df.groupby("Continent")['life_expectancy'].median().plot(kind = 'barh')
ax.set_ylabel("")

# Initialize the plot first, take the ax
fig, ax = plt.subplots(figsize=(10,5))

# Pass the ax to the  .plot function
df.groupby("Continent")['life_expectancy'].median().plot(kind = 'barh')
ax.set_ylabel("")
ax.grid(True)

# Initialize the plot first, take the ax
fig, ax = plt.subplots(figsize=(7,3))

# Pass the ax to the  .plot function
df.groupby("Continent")['life_expectancy'].median().plot(kind = 'barh')
ax.set_ylabel("")

# When plot the grid, you can send it options(use ? )
ax.grid(color='MidnightBlue', linestyle=':', linewidth=0.5)
# Erase the line inside the barh
ax.set_axisbelow(True)

# get rid of the top and right lines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


# color : google "html colors"
# linestyle: : -. -- 

# Initialize the plot first, take the ax
fig, ax = plt.subplots(figsize=(7,3))

# Pass the ax to the  .plot function
df.groupby("Continent")['life_expectancy'].median().plot(kind = 'barh')
ax.set_ylabel("")

# When plot the grid, you can send it options(use ? )
ax.grid(color='MidnightBlue', linestyle=':', linewidth=0.5)
# Erase the line inside the barh
ax.set_axisbelow(True)

# get rid of the lines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

plt.tick_params(
which ='major',
top = 'off',
left = 'off',
right = 'off',
bottom = 'off',
labeltop='on',
labelbottom='on')

# color : google "html colors"
# linestyle: : -. -- 

# Initialize the plot first, take the ax
fig, ax = plt.subplots(figsize=(7,3))

# Pass the ax to the  .plot function
df.groupby("Continent")['life_expectancy'].median().plot(kind = 'barh')
ax.set_ylabel("")

# ax.xaxis.grid takes all the same options as ax.grid
ax.xaxis.grid(color='MidnightBlue', linestyle=':', linewidth=0.5)
ax.yaxis.grid(color='Pink', linestyle='-.', linewidth=0.5)

# Erase the line inside the barh
ax.set_axisbelow(True)

# get rid of the lines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

plt.tick_params(
which ='major',
top = 'off',
left = 'off',
right = 'off',
bottom = 'off',
labeltop='on',
labelbottom='on')

# color : google "html colors"
# linestyle: : -. -- 

ax.set_xlim((0,79))

# Initialize the plot first, take the ax
fig, ax = plt.subplots(figsize=(8,5))

# Pass the ax to the  .plot function
df.groupby("Continent")['life_expectancy'].median().plot(kind = 'barh')
ax.set_ylabel("")

# ax.xaxis.grid takes all the same options as ax.grid
ax.xaxis.grid(which ='major', color='MidnightBlue', linestyle=':', linewidth=1)
ax.yaxis.grid(which='minor', color='Pink', linestyle='-.', linewidth=0.5)

# Turn on a minor
ax.minorticks_on()

# Erase the line inside the barh
ax.set_axisbelow(True)

# get rid of the lines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

plt.tick_params(
which ='major',
top = 'off',
left = 'off',
right = 'off',
bottom = 'off',
labeltop='on',
labelbottom='on')

plt.tick_params(
which ='minor',
top = 'off',
left = 'off',
right = 'off',
bottom = 'off',
labeltop='off',
labelbottom='off')

# color : google "html colors"
# linestyle: : -. -- 

median = df['life_expectancy'].median()
ax.plot([median,median], [-1,10], c='red', linestyle='-', linewidth=0.5)

ax.set_xlim((0,79))

df['life_expectancy'].median()

# Initialize the plot first, take the ax
fig, ax = plt.subplots(figsize=(8,5))

# Pass the ax to the  .plot function
df.groupby("Continent")['life_expectancy'].median().plot(kind = 'barh')
ax.set_ylabel("")

# ax.xaxis.grid takes all the same options as ax.grid
ax.xaxis.grid(which ='major', color='MidnightBlue', linestyle=':', linewidth=1)
ax.yaxis.grid(which='minor', color='Pink', linestyle='-.', linewidth=0.5)

# get rid of this line 
#ax.minorticks_on()

# Erase the line inside the barh
ax.set_axisbelow(True)

# get rid of the lines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

plt.tick_params(
which ='major',
top = 'off',
left = 'off',
right = 'off',
bottom = 'off',
labeltop='on',
labelbottom='on')

plt.tick_params(
which ='minor',
top = 'off',
left = 'off',
right = 'off',
bottom = 'off',
labeltop='off',
labelbottom='off')

# color : google "html colors"
# linestyle: : -. -- 

median = df['life_expectancy'].median()
ax.plot([median,median], [-1,10], c='red', linestyle='-', linewidth=0.5)

ax.annotate(s="Median life expectancy, 70 years", xy=(71, 0), color='red')

ax.set_xlim((0,79))

# Initialize the plot first, take the ax
fig, ax = plt.subplots(figsize=(8,5))

# Pass the ax to the  .plot function
df.groupby("Continent")['life_expectancy'].median().plot(kind = 'barh', ax=ax)
ax.set_ylabel("")

# ax.xaxis.grid takes all the same options as ax.grid
ax.xaxis.grid(which ='major', color='MidnightBlue', linestyle=':', linewidth=1)
ax.yaxis.grid(which='minor', color='Pink', linestyle='-.', linewidth=0.5)

# get rid of this line 
#ax.minorticks_on()

# Erase the line inside the barh
ax.set_axisbelow(True)

# get rid of the lines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

plt.tick_params(
which ='major',
top = 'off',
left = 'off',
right = 'off',
bottom = 'off',
labeltop='on',
labelbottom='on')

plt.tick_params(
which ='minor',
top = 'off',
left = 'off',
right = 'off',
bottom = 'off',
labeltop='off',
labelbottom='off')

# color : google "html colors"
# linestyle: : -. -- 

median = df['life_expectancy'].median()
ax.plot([median,median], [-1,10], c='red', linestyle='-', linewidth=0.5)

ax.annotate(s="Median life expectancy, 70 years", xy=(71, 0), color='red')

ax.set_xlim((30,79))

plt.savefig('life.pdf', transparent=True)

# Fivethirtyeight scatterplot of countries'GDP and life expecancy
fig, ax = plt.subplots()

df.plot(kind='scatter', x = 'GDP_per_capita', y = 'life_expectancy', ax = ax)

df[df['Continent']=='Asia'].plot(color='red', kind='scatter', x = 'GDP_per_capita', y = 'life_expectancy', ax = ax)
df[df['Continent']=='Africa'].plot(color='blue', kind='scatter', x = 'GDP_per_capita', y = 'life_expectancy', ax = ax)
df[df['Continent']=='Oceania'].plot(color='green', kind='scatter', x = 'GDP_per_capita', y = 'life_expectancy', ax = ax)

fig, ax = plt.subplots(figsize=(10,8))
fig.set_facecolor('lightgray')
ax.set_axis_bgcolor('lightgray')

ax.set_prop_cycle('color',['pink','purple','red','deeppink','lightcoral','magenta'])

ax.set_title("GDP and Life Expectancy Over The World")
ax.set_ylabel("Life Expectancy(year)")
ax.set_xlabel('GDP per capita($)')

ax.grid(linestyle="-", color='white')
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)

plt.tick_params(
which ='major',
top = 'off',
left = 'off',
right = 'off',
bottom = 'off',
labeltop='off',
labelbottom='on')


for continent, selection in df.groupby("Continent"):
    ax.plot(selection["GDP_per_capita"], selection['life_expectancy'], label=continent, marker='o', linestyle="", markeredgewidth=0)
    
ax.legend(loc="lower right")   
ax.set_ylim((30,85))

fig, ax = plt.subplots(figsize=(10,8))
fig.set_facecolor('lightgray')
ax.set_axis_bgcolor('lightgray')

ax.set_prop_cycle('color',['skyblue','POWDERBLUE','dodgerblue','ROYALBLUE','mediumslateblue','navy'])

ax.set_title("GDP and Life Expectancy Over The World")
ax.set_ylabel("Life Expectancy(year)")
ax.set_xlabel('GDP per capita($)')

ax.grid(linestyle="-", color='white')
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)

plt.tick_params(
which ='major',
top = 'off',
left = 'off',
right = 'off',
bottom = 'off',
labeltop='off',
labelbottom='on')


for continent, selection in df.groupby("Continent"):
    ax.plot(selection["GDP_per_capita"], selection['life_expectancy'], label=continent, marker='o', linestyle="", markeredgewidth=0)
    
ax.legend(loc="lower right")   
ax.set_ylim((30,85))



