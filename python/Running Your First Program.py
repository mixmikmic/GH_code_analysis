
import pickle
import pandas as pd

# Handy list of the different types of encodings
encoding = ['latin1', 'iso8859-1', 'utf-8'][1]

def load_object(obj_name):
    pickle_path = '../../saves/pickle/' + obj_name + '.pickle'
    try:
        object = pd.read_pickle(pickle_path)
    except:
        with open(pickle_path, 'rb') as handle:
            object = pickle.load(handle)
    
    return(object)

def save_dataframes(**kwargs):
    csv_folder = '../../saves/csv/'
    for frame_name in kwargs:
        csv_path = csv_folder + frame_name + '.csv'
        kwargs[frame_name].to_csv(csv_path, sep=',', encoding=encoding, index=False)

# Classes, functions, and methods cannot be pickled
def store_objects(**kwargs):
    for obj_name in kwargs:
        if hasattr(kwargs[obj_name], '__call__'):
            raise RuntimeError('Functions cannot be pickled.')
        obj_path = '../../saves/pickle/' + str(obj_name)
        pickle_path = obj_path + '.pickle'
        if isinstance(kwargs[obj_name], pd.DataFrame):
            kwargs[obj_name].to_pickle(pickle_path)
        else:
            with open(pickle_path, 'wb') as handle:
                pickle.dump(kwargs[obj_name], handle, pickle.HIGHEST_PROTOCOL)


import re
import numpy as np

# Create a dataframe of the Jewish population data from
# https://en.wikipedia.org/wiki/Historical_Jewish_population_comparisons
file_path = '../../data/html/JewishPopulation.html'
jews_df = pd.read_html(file_path)[0]
numeric_columns = ['Pop1900', 'Pct1900', 'Pop1942', 'Pct1942',
                  'Pop1970', 'Pct1970', 'Pop2010', 'Pct2010']
int_columns = ['Pop1900', 'Pop1942', 'Pop1970', 'Pop2010']
float_columns = ['Pct1900', 'Pct1942', 'Pct1970', 'Pct2010']
jews_df.columns = ['Region'] + numeric_columns

def f(element):
    
    return str(element).split('[')[0]

jews_df['Region'] = jews_df['Region'].map(f)
jews_df['Pop1900'] = jews_df['Pop1900'].map(f)
num_regex = re.compile(r'[^0-9.]+')

def f(row):
    for column_name in numeric_columns:
        row[column_name] = num_regex.sub('', str(row[column_name]))
        try:
            row[column_name] = float(row[column_name])
        except Exception:
            row[column_name] = np.NaN
    
    return row
        
jews_df = jews_df.apply(f, axis=1)
jews_df.set_index(keys='Region', inplace=True)

# Create dataframes from the GapMinder data from
# http://www.gapminder.org/data/

encoding = ['latin1', 'iso8859-1', 'utf-8'][1]

income_df = pd.read_csv('../../data/csv/income_df.csv', encoding=encoding)
income_df.set_index(keys='Country', inplace=True)

population_df = pd.read_csv('../../data/csv/population_df.csv', encoding=encoding)
population_df.set_index(keys='Country', inplace=True)


# Regions that can't be compared directly to the income dataframe
# (basically lots of them)
jews_df.index.difference(income_df.index)


from difflib import SequenceMatcher

def similar(a, b):
    return SequenceMatcher(None, str(a), str(b)).ratio()

#Check the closest names for typos
def check_for_typos(first_list, second_list):
    rows_list = []
    for first_item in first_list:
        max_similarity = 0.0
        max_item = first_item
        for second_item in second_list:
            this_similarity = similar(first_item, second_item)
            if this_similarity > max_similarity:
                max_similarity = this_similarity
                max_item = second_item

        # Get input row in dictionary format; key = col_name
        row_dict = {}
        row_dict['first_item'] = first_item
        row_dict['second_item'] = max_item
        row_dict['max_similarity'] = max_similarity

        rows_list.append(row_dict)

    column_list = ['first_item', 'second_item', 'max_similarity']
    name_similarities_df = pd.DataFrame(rows_list, columns=column_list)
    
    return name_similarities_df


# Closest regions to the countries in the income dataframe that are not paired up
name_similarities_df = check_for_typos(jews_df.index.difference(income_df.index), income_df.index)
name_similarities_df.sort_values(['max_similarity'], ascending=False).head(10)


# Get the intersection of the jew and income indexes
jews_df.index.intersection(income_df.index)


# Create a Jewish population dataframe with the common rows
jew_pop_df = jews_df.loc[jews_df.index.isin(jews_df.index.intersection(income_df.index).tolist()),
                         int_columns]
jew_pop_df.columns = [n[3:] for n in int_columns]
jew_pop_df.index.rename('Country', inplace=True)
jew_pop_df


# Assume all Jews left Algeria and Libya by 2010
jew_pop_df['2010'].fillna(0, inplace=True)

# Interpolate the rest of the missing values
jew_pop_df.columns = [pd.datetime(int(n[3:]), 1, 1) for n in int_columns]
jew_pop_df.interpolate(method='time', axis=1, inplace=True)
jew_pop_df.columns = [n[3:] for n in int_columns]
jew_pop_df


# Frequency tables are kind of pointless,
# as the data has mostly unique values
jew_pop_df['1900'].value_counts().to_frame().head(5)
jew_pop_df['1942'].value_counts().to_frame().head(5)
jew_pop_df['1970'].value_counts().to_frame().head(5)
jew_pop_df['2010'].value_counts().to_frame().head(5)


# Create a personal income dataframe with the common rows
pers_inc_df = income_df.loc[jews_df.index.intersection(income_df.index), [n[3:] for n in int_columns]]
pers_inc_df.index.rename('Country', inplace=True)
pers_inc_df


# Frequency tables are kind of pointless,
# as the data has mostly unique values
pers_inc_df['1900'].value_counts().to_frame().head(5)
pers_inc_df['1942'].value_counts().to_frame().head(5)
pers_inc_df['1970'].value_counts().to_frame().head(5)
pers_inc_df['2010'].value_counts().to_frame().head(5)


# Histograms are better
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(15, 5))
num_rows = 2
num_cols = 4
i = 1
for row_num, df in enumerate([pers_inc_df, jew_pop_df]):
    for col_num, col_name in enumerate([n[3:] for n in int_columns]):
        ax = fig.add_subplot(num_rows, num_cols, i, autoscale_on=True)
        i += 1
        note_text = col_name
        if row_num > 0:
            xrot = 45
            note_text += '\nJewish Population'
        else:
            xrot = 0
            note_text += '\nPersonal Income'
        note = ax.text(0.65, 0.97, note_text, transform=ax.transAxes, fontsize=14,
                       fontweight='normal', va='top', ha='center', alpha=.6)
        histogram = df[col_name].hist(ax=ax, grid=False, bins=10, xrot=xrot)


save_dataframes(pers_inc_df=pers_inc_df, jew_pop_df=jew_pop_df)
data_df = jew_pop_df.join(pers_inc_df, lsuffix='_jew_pop', rsuffix='_pers_inc')
data_df = data_df.applymap(lambda x: int(x))
save_dataframes(data_df=data_df)
store_objects(data_df=data_df)
data_df


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(111, autoscale_on=True)
x = data_df['1900_jew_pop']
y = data_df['1900_pers_inc']
dots = ax.scatter(x, y)


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(111, autoscale_on=True)
x = data_df['1942_jew_pop']
y = data_df['1942_pers_inc']
dots = ax.scatter(x, y)


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(111, autoscale_on=True)
x = data_df['1970_jew_pop']
y = data_df['1970_pers_inc']
dots = ax.scatter(x, y)


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(111, autoscale_on=True)
x = data_df['2010_jew_pop']
y = data_df['2010_pers_inc']
dots = ax.scatter(x, y)


from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

def f(row):
    y_jew_pop = [row['1900_jew_pop'], row['1942_jew_pop'], row['1970_jew_pop'], row['2010_jew_pop']]
    y_pers_inc = [row['1900_pers_inc'], row['1942_pers_inc'], row['1970_pers_inc'], row['2010_pers_inc']]
    lines = ax.plot(row[jew_pop_columns], row[pers_inc_columns], label=row.index)
    
    return r2_score(y_jew_pop, y_pers_inc)

fig = plt.figure(figsize=(15, 10), dpi=80)
ax = fig.add_subplot(111, autoscale_on=True)
data_df['coefficient_of_dermination'] = data_df.apply(f, axis=1)
data_df.sort_values('coefficient_of_dermination')['coefficient_of_dermination']

