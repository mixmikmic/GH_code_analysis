import pandas as pd
import numpy as np
from src.utils import import_data
from src.topic_modelling import topic_modelling
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
get_ipython().run_line_magic('matplotlib', 'inline')

df = import_data("job-hunter-plus-data", "indeed_data.csv")
df2 = df #Make a copy to avoid having to download from the S3 bucket one every run.

print(df["job_description"][1])

topic_modelling(df, "San+Francisco", n_topics=15)

sf_topics = {"San Francisco": ["Finance", "Veterans", "General", "SalesForce", "Data Science",             "EEO", "HR", "Clinical Research", "Digital Marketing", "General", "Data Science",             "Qualifications", "General", "General", "General"]}

topic_modelling(df, "New+York", n_topics=15)

ny_topics = {"New York": ["Business Analysis", "Medical", "Veterans", "General", "Medical",             "General", "Data Science", "Finance", "Medical", "Digital Marketing",             "General", "EEO", "Clinical Research", "Finance", "General"]}

topic_modelling(df, "Chicago", n_topics=15)

ch_topics = {"Chicago": ["Technical Support", "General", "General", "General", "Veterans", "Data Science",             "Intellectual Property", "EEO", "General", "Medical", "General", "Digital Marketing",             "Finance", "Consulting", "General"]}

topic_modelling(df, "Austin", n_topics = 15)

au_topics = {"Austin":["Business Analysis", "General", "Qualifications", "HR", "General", "Veterans",             "Digital Marketing", "Data Science", "General", "InfoSec", "EEO", "General",             "Sales", "Digital Marketing", "General"]}
list(au_topics.values())[0]

list1 = []
list2 = []
for i, topics in enumerate([au_topics, ch_topics, ny_topics, sf_topics]):
    for k, v in topics.items():
        for j, value in enumerate(v):
            if value == "General":
                continue
            list1 += [k]
            list2 += [value]
arr = np.column_stack((list1, list2))

plt.subplots(figsize=[16, 10])
rc={'axes.labelsize': 20, 'font.size': 30 , 'legend.fontsize': 12.0}
plt.rcParams.update(**rc)

plt.rc('font', **font)
sns.set(rc=rc)
sns.countplot(y = arr[:,0], hue = arr[:,1])
plt.tick_params(labelsize=16)
plt.xticks([0, 1, 2, 3])
plt.xlabel("Number of Topic Appearances in Top 15")
plt.savefig('topic_barcodes.png', bbox_inches='tight')

arr2 = arr[(arr[:,0] == "San Francisco") | (arr[:,0] == "New York")]
plt.subplots(figsize=[8,9])
rc={'axes.labelsize': 20, 'font.size': 30 , 'legend.fontsize': 16.0}
plt.rcParams.update(**rc)

plt.rc('font', **font)
sns.set(rc=rc)
order = ["Business Analysis", "Medical", "Finance", "Veterans", "Digital Marketing", "EEO",         "Clinical Research", "Data Science", "SalesForce", "HR", "Qualifications"]
sns.countplot(y = arr2[:,1], hue = arr2[:,0], order=order)
plt.tick_params(labelsize=16)
plt.xticks([0, 1, 2, 3])
plt.xlabel("Number of Topic Appearances in Top 15")
plt.savefig('two_city_comparison.png', bbox_inches='tight')

