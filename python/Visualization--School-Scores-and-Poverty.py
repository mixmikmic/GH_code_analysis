get_ipython().magic('matplotlib inline')

from os.path import join
import matplotlib.pyplot as plt
import pandas as pd
DATA_DIR = join('data', 'schools', 'extracted')
frpm_dataname = join(DATA_DIR, 'frpm-2014.csv')
sat_dataname = join(DATA_DIR, 'sat-2014.csv')

frpm_df = pd.read_csv(frpm_dataname, na_values=['*'])
sat_df = pd.read_csv(sat_dataname, na_values=['*'])

# Filtering the SAT schools
sat_df = sat_df[sat_df['number_of_test_takers'] >= 20] 

plt.scatter(sat_df['avg_reading_score'], 
            sat_df['avg_writing_score'])
plt.xlabel('Average reading score')
plt.ylabel('Average writing score');

plt.scatter(sat_df['avg_writing_score'], 
            sat_df['avg_math_score'])
plt.xlabel('Average writing score')
plt.ylabel('Average math score');

# merging the dataframes
xdf = pd.merge(left=sat_df, right=frpm_df, on='cds')

fig, ax = plt.subplots()
ax.scatter(xdf['adjusted_pct_eligible_frpm_k12'] * 100, 
            xdf['percent_scores_gte_1500'])
ax.set_xlabel('% eligible for free or reduced-price lunch')
ax.set_ylabel('% SAT scores >= 1500')
ax.set_xlim(xmin=0, xmax=100)
ax.set_ylim(ymin=0, ymax=100);

# adding a trendline
import numpy as np
xvals = xdf['adjusted_pct_eligible_frpm_k12'] * 100
yvals = xdf['percent_scores_gte_1500']
plt.scatter(xvals, yvals)
plt.xlabel('% eligible for free or reduced-price lunch')
plt.ylabel('% SAT scores >= 1500');

z = np.polyfit(xvals, yvals, 1)
p = np.poly1d(z)
plt.plot(xvals, p(xvals), color='yellow', linewidth=3);

fig, ax = plt.subplots()
ax.hist(xdf['percent_scores_gte_1500'], bins=20, color='gray');
ax.set_xlabel('Percentage of scores >= 1500')
ax.set_ylabel('Number of schools')
ax.set_title('Histogram, average SAT scores per school, 2014');

fig, ax = plt.subplots()
ax.hist(xdf['adjusted_pct_eligible_frpm_k12'], bins=10, color='orange');
ax.set_xlabel('Percentage of students eligible for FRPM')
ax.set_ylabel('Number of schools')
ax.set_title('Histogram, 2014');


fig, ax = plt.subplots()
richdf = xdf[xdf['adjusted_pct_eligible_frpm_k12'] < 0.1]
poordf = xdf[xdf['adjusted_pct_eligible_frpm_k12'] >= 0.1]
datalist = [richdf['percent_scores_gte_1500'], poordf['percent_scores_gte_1500']]

ax.hist(datalist, bins=20, 
         stacked=True, color=['#99FF99', 'gray'])
ax.set_xlabel('Percentage of scores >= 1500')
ax.set_ylabel('Number of schools');
ax.legend(['< 10% FRPM', '10%+ FRPM']);



