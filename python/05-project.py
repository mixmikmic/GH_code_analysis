# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import os

# Plot settings
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')
fontsize = 20 # size for x and y ticks
plt.rcParams['legend.fontsize'] = fontsize
plt.rcParams.update({'font.size': fontsize})

# Connect to the database
fn = os.path.join('data','eicu_demo.sqlite3')
con = sqlite3.connect(fn)
cur = con.cursor()

admissiondx = pd.read_sql_query("select * from admissiondx", con)
apacheapsvar = pd.read_sql_query("select * from apacheapsvar", con)
apachepatientresult = pd.read_sql_query("select * from apachepatientresult", con)
apachepredvar = pd.read_sql_query("select * from apachepredvar", con)
careplancareprovider = pd.read_sql_query("select * from careplancareprovider", con)
careplaneol = pd.read_sql_query("select * from careplaneol", con)
careplangeneral = pd.read_sql_query("select * from careplangeneral", con)
careplangoal = pd.read_sql_query("select * from careplangoal", con)
careplaninfectiousdisease = pd.read_sql_query("select * from careplaninfectiousdisease", con)
diagnosis = pd.read_sql_query("select * from diagnosis", con)
lab = pd.read_sql_query("select * from lab", con)
pasthistory = pd.read_sql_query("select * from pasthistory", con)
patient = pd.read_sql_query("select * from patient", con)
treatment = pd.read_sql_query("select * from treatment", con)
vitalaperiodic = pd.read_sql_query("select * from vitalaperiodic", con)
vitalperiodic = pd.read_sql_query("select * from vitalperiodic", con)

