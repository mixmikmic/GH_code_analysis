import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')
import seaborn as sns

# Read in data
trials = pd.read_csv('assets/study_fields.csv')

# split pipe delimited sponsor names into a list in each cell
s = trials['Sponsor/Collaborators'].str.split('|')

# The lead sponsor is the first one listed - generate new list with only lead sponsor
lead_sponsors = [row[0] for row in s]

# Turn lead_sponsors list to a pandas series
lead_sponsors_series = pd.Series(lead_sponsors)

# Get value counts
lead_sponsors_series.value_counts().sort_values(ascending=False)[:50]

# Number of cancer sites - for number of bars on plot
num_sponsors = np.arange(20)

# Trial totals - for length of bars
trial_totals_by_sponsor = lead_sponsors_series.value_counts().sort_values(ascending=False)[:20].values

# Names of cancer sites - for bar labels
sponsor_names = lead_sponsors_series.value_counts().sort_values(ascending=False)[:20].index

# Create horizontal bar
plt.barh(num_sponsors, trial_totals_by_sponsor, align='center', alpha=0.4)

# Create yticks
plt.yticks(num_sponsors, sponsor_names)

# Create xlabel
plt.xlabel('Number of Trials')

# Invert graph
plt.gca().invert_yaxis()

# Title
plt.title('Top Twenty Trial Sponsors')



