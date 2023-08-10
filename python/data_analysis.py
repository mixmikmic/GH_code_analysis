import pandas as pd

def count_unique(s):
    values = s.unique()
    return sum(1 for v in values if pd.notnull(v))

def missing_pct(s):
    missing = N - s.count()
    return missing * 100.0 / N

def complete_pct(s):
    return 100 - missing_pct(s)

def summarize_completeness_uniqueness(df):
    rows = []
    for col in df.columns:
        rows.append([col, '%.0f%%' % complete_pct(df[col]), count_unique(df[col])])
    return pd.DataFrame(rows, columns=['Column Name', 'Complete (%)','Unique Values'])

def summarize_completeness_over_time(df, time_col, transpose=True):
    x = df.groupby(time_col).count()
    x = x.div(df.groupby(time_col).size(), axis=0)
    for col in x.columns:
        x[col] = x[col].apply(lambda value: '%.0f%%' % (value * 100))
    if transpose:
        return x.T
    return x

files = ['San_Francisco_City_Survey_Data_1996-2015.csv', 'Park_Evaluation_Scores_starting_Fiscal_Year_2015.csv',
'Paving_PCI_Scores_Historical_Data.csv', 'San_Francisco_Analysis_Neighborhoods.csv']

survey_df = pd.read_csv('San_Francisco_City_Survey_Data_1996-2015.csv')
N = len(survey_df)
for col in survey_df.columns:
    if 'Date' in col:
        df[col] = pd.to_datetime(survey_df[col], format="%m/%d/%Y")
print survey_df.shape
survey_data_summary = summarize_completeness_over_time(survey_df, 'year')
survey_data_summary

get_ipython().magic('pylab inline')
survey_df['zipcode'].dropna().unique().shape

park_eval_df = pd.read_csv('Park_Evaluation_Scores_starting_Fiscal_Year_2015.csv')
N = len(park_eval_df)
summarize_completeness_uniqueness(park_eval_df)

paving_pci_df = pd.read_csv('Paving_PCI_Scores_Historical_Data.csv')
N = len(paving_pci_df)
summarize_completeness_uniqueness(paving_pci_df)

neighborhoods_df = pd.read_csv('San_Francisco_Analysis_Neighborhoods.csv')
N = len(neighborhoods_df)
summarize_completeness_uniqueness(neighborhoods_df)

neighborhoods_df



