import pandas as pd

df = pd.read_csv('datasets/dataset.csv')

df.education.unique()

#school, college, bachelor, master
df['education'] = df['education'].replace(['High School or Below'], 'school')
df['education'] = df['education'].replace(['Bechalor'], 'bachelor')
df['education'] = df['education'].replace(['Master or Above'], 'master')

df.to_csv('datasets/dataset.csv')
df[:10]

df = df.drop(['loan_id', 'effective_date', 'due_date', 'paid_off_time', 'past_due_days'], axis = 1)

# The dataframe holds the needed columns now. Cool.
df[:10]

df.info()

# Lets clean the data and create columns if needed.
df['Gender'].unique()

df_sex = pd.get_dummies(df['Gender'])

df_sex[:10]

df = pd.concat([df,df_sex] , axis=1)

df[:10]

# Now drop the gender column from the main df and add df_sex to df

df = df.drop(['Gender'], axis=1)

df[:10]

# Similary lets do the same process for both load_status and education.
# This process is called Categorical Conversion into Numerics of One-hot-coding

df_loan_status = pd.get_dummies(df['loan_status'])
df_education = pd.get_dummies(df['education'])
df = pd.concat([df, df_loan_status], axis=1)
df = pd.concat([df, df_education], axis=1)
df = df.drop(['loan_status', 'education'], axis=1)

df[:10]

df.info()

df_to_norm = df[['Principal', 'terms', 'age']]
df_to_norm[:10]

df_norm = (df_to_norm - df_to_norm.min()) / (df_to_norm.max() - df_to_norm.min())
df_norm[:10]

df = df.drop(['Principal', 'terms', 'age'], axis=1)
df = pd.concat([df,df_norm], axis=1)

df[:10]

