# Tell Jupyter to display matplotlib plots in your notebook
get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import matplotlib.pyplot as plt

# Let's resize all our charts to be a bit bigger
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12, 8)

# matplotlib has certain default styles that you can specify at the top of your notebook
# https://matplotlib.org/users/style_sheets.html
# Here, I use the 'bmh' style - just a personal preference!
plt.style.use('bmh')

# Loading the Greek life data
df = pd.read_csv('data/duke-greek-life.csv')
print(df.shape)
df.head()

# How many students are there?
len(df)

# How many students are in a Greek organization?
df['Greek Organization'].value_counts()

# How many Greek orgs are there?
df['Greek Organization'].nunique() - 1

# What percentage of students are in a Greek org?
len(df[df['Greek Organization'] != "None"]) / len(df)

# What's the distribution of high school tuitions?
df['Tuition of High School'].describe()

pd.to_numeric(df['Tuition of High School'])

df['Tuition of High School'].replace(to_replace="16,600", value="16600", inplace=True)
pd.to_numeric(df['Tuition of High School'])

pd.to_numeric(df['Tuition of High School'].str.replace(',',''))

df['Tuition of High School'] = pd.to_numeric(df['Tuition of High School'].str.replace(',',''))
df['Tuition of High School'].describe()

df['Tuition of High School'].hist()

df['Tuition of High School'].isnull().value_counts()

df['Tuition of High School'].hist(bins=20)
plt.axvline(df['Tuition of High School'].mean(), 
            color='yellow', 
            label=f"Mean: ${df['Tuition of High School'].mean():,.2f}")
plt.legend()
plt.title('Distribution of high school tuitions\n(Duke University students)')
plt.xlabel('USD ($)')
plt.ylabel('Frequency')
plt.show()

# Let's one hot encode (AKA make dummy variables) the public or private high school column
df['Public or Private High School'].value_counts()

school_ohe = pd.get_dummies(df['Public or Private High School'], prefix='school_type')
school_ohe.head()

# Let's add those dummies back into the original dataframe via merge!
df = pd.merge(df, school_ohe, left_index=True, right_index=True)
df.head()

# I've forgotten - which columns do we have?
df.columns

df['Greek Council'].value_counts() / len(df)

# I don't really like typing out the long Proper English Words as column names,
# let's make them lowercase snake_case!
for col in df.columns:
    df.rename(columns={col: col.lower().replace(' ', '_') }, inplace=True)
    
# Check to make sure it worked    
df.head()

from IPython.display import YouTubeVideo
YouTubeVideo('P_q0tkYqvSk')

from scipy.stats import bernoulli

# This function will take each row as the input
def guessing_sex(row):
    """Male = 1, female = 0
    """
    if row['greek_council'] == 'Fraternity':
        return 1
    elif row['greek_council'] == 'Sorority':
        return 0
    else:
        return bernoulli.rvs(p=0.5)
    
# Applying that function to the dataframe
df['guessed_sex'] = df.apply(guessing_sex, axis=1)
df['guessed_sex'].value_counts() / len(df)

# Sanity check
for membership in df['greek_council'].unique():
    print(f"\nNow checking: {membership}")
    print(df[df['greek_council']==membership]['guessed_sex'].value_counts() / len(df))

