df = pd.read_csv('Data/co_rev_sample_filtered.csv', encoding = 'utf-8')

review_spec_cols = ['agg_rating','comp_ben_rating', 
    'culture_rating', 'jobsec_advancement_rating', 
       'management_rating', 'work_life_rating', 'helpful_downvote_count',
       'helpful_upvote_count', 'review_id','review_date', 'review_title',
       'reviewer_company_empl_status', 'reviewer_job_location',
       'reviewer_job_title', 'main_text', 'con_text', 'pro_text'
]

company_cols = df.columns[~(df.columns.isin(review_spec_cols))]

numeric_review_spec_cols = df[review_spec_cols].describe().columns
df[review_spec_cols].describe()

from itertools import compress
review_labels = list(compress(review_spec_cols,[(x not in numeric_review_spec_cols) for x in review_spec_cols]))
review_labels

plt.hist(df['agg_rating'], bins=20, color="#5ee3ff")

get_ipython().run_line_magic('matplotlib', 'inline')

plt.hist(df['agg_rating'], color = "#0033cc")
plt.ylabel('count')
plt.xlabel('agg_rating')

plt.scatter(df['company_overall_rating'], df['agg_rating'])
plt.xlabel('company_overall_rating')
plt.ylabel('agg_rating')

df_grp = df.copy()
df_grp = df_grp.groupby('company_name')
df_grp['agg_rating'].mean()

sns.kdeplot(df_grp['agg_rating'].mean(), shade=True, label='Estimated PDF of company mean agg_rating score')

sns.distplot(df_grp['agg_rating'].mean())

sns.kdeplot(df_grp['company_overall_rating'].mean(), shade=True, label='Estimated PDF of company mean agg_rating score')

sns.distplot(df_grp['company_overall_rating'].mean())

