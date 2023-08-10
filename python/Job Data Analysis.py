import pandas as pd
get_ipython().magic('matplotlib inline')

job_data = pd.read_csv("../web-scraping/jobs-data.csv")
job_data.head()

job_data.dtypes

job_data.shape

job_data['job_title'].value_counts(ascending=False)

job_data['company_name'].value_counts(ascending=False)

job_data['job_location'].value_counts(ascending=False)

job_data['state'] = job_data['job_location'].str.extract(', (\w{2})', expand=False)
job_data.head()

ax = job_data['state'].value_counts(ascending=True).plot(kind="barh", figsize=(10,10), xlim=(0,450))
# add counts as annotations
# http://stackoverflow.com/questions/23591254/python-pandas-matplotlib-annotating-labels-above-bar-chart-columns
for p in ax.patches:
    ax.annotate("%d" % p.get_width(), (p.get_x() + p.get_width(), p.get_y()), xytext=(0, 0), textcoords='offset points')

job_data[job_data['job_location'].str.contains("Pittsburgh")]

company_data = pd.read_csv("company-data.csv")
company_data.head()

company_data.dtypes

company_data.shape

company_data.describe()

company_data['overall_rating'].hist()

company_data['culture_rating'].hist()

company_data['compensation_benefits_rating'].hist()

company_data['management_rating'].hist()

company_data['js_advancement_rating'].hist()



