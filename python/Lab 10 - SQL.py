# Everything based on Class Slides ("More SQL and Python Integration")

import sqlite3
import pandas as pd
from matplotlib import pyplot as plt

con = sqlite3.connect('atus.sqlite')

# To read the script directly from the q4.sql file your created in the folder
with open('q4.sql') as f:
    sql = f.read()

# Run the SQL query and store result in a Pandas DataFrame
df = pd.read_sql(sql , con)

# Remap the 'edited_sex' column to the true values
df['sex'] = df['sex'].map({1:'Male', 2:'Female'})

# Use Pandas set_index, unstack, plot



