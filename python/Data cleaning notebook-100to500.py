import numpy as np
import pandas as pd
import re

file = "indeed_scrapy/scrape_data/company_reviews100to500fort20171020.csv"
df = pd.read_csv(file)

print(df.shape)
df.head()

company_bar_score_names = ["Quick-paced", 'Slow-paced', 'Stressful', 'Balanced', 'Competitive', 'Relaxed', 'Cut-throat']
bar_dict = {'bar_name1' : 'bar_pct1', 
            'bar_name2' : 'bar_pct2', 
            'bar_name3' : 'bar_pct3', 
            'bar_name4' : 'bar_pct4', 
            'bar_name5' : 'bar_pct5'
}

company_rate_names = ["Compensation/Benefits", "Culture", "Job Security/Advancement", "Work/Life Balance", "Management"]
rate_dict = {'rating_name1' : 'rating_scr1', 
            'rating_name2' : 'rating_scr2', 
            'rating_name3' : 'rating_scr3', 
            'rating_name4' : 'rating_scr4', 
            'rating_name5' : 'rating_scr5'
}  

df['review_id'] = df.index + 127000
df['review_id'].head()

test = bar_dict.keys()
type(bar_dict)

#testing for loop
test = df #need to add .copy() to test without changing df
i = 0
mask = test['bar_name2'] == company_bar_score_names[i]

test.loc[mask,company_bar_score_names[i]] = test.loc[mask,'bar_pct2']
print(company_bar_score_names[i])
test[company_bar_score_names[i]].head()

df.filter(regex = '^b').head()

#bar names
for name in company_bar_score_names:
    for bar, bar_pct in bar_dict.items():
        mask = df[bar].str.strip() == name
        df.loc[mask, name] = df.loc[mask, bar_pct]
        
    

#rating names
for name in company_rate_names:
    for rat, rat_scr in rate_dict.items():
        mask = df[rat].str.strip() == name
        df.loc[mask, name] = df.loc[mask, rat_scr]


df.iloc[:, -13:].head()

df.iloc[:,-12:].isnull().apply(pd.value_counts)

def string_pct_to_float(strpct):
    try:
        return float(strpct.strip('%'))/100
    except:
        return strpct

def parse_widths_to_rating(string):
    
    try:
        result = string_pct_to_float(string.replace("width: ","").replace(";","").strip())
        if result == 0:
            return np.nan
        else:
            return result * 5
    except:
        return string

print(string_pct_to_float('80%'))
print(string_pct_to_float('100%'))
print(string_pct_to_float(123))
print(parse_widths_to_rating('width: 100.0%'))
print(parse_widths_to_rating(123))
print(parse_widths_to_rating('width: 100.0%'))

#replace widths with 
widths_to_replace_mask = df.iloc[0,:].str.contains(r"^width: ") 

cols = df.columns[widths_to_replace_mask.fillna(False)]
cols

df[cols] = df[cols].applymap(parse_widths_to_rating)

df[cols].head()

bar_names_count = df[list(bar_dict.keys())].apply(pd.Series.value_counts)
bar_names_count['totals'] = bar_names_count.sum(axis = 1)
bar_names_count = bar_names_count.append(pd.Series(bar_names_count.sum(axis=0), name = 'subtotals'))
bar_names_count

val_cols = bar_names_count.index[:7]
#from above: df.filter(regex="^rat").isnull().apply(pd.value_counts)
val_col_counts = df[val_cols].isnull().apply(pd.value_counts)
val_col_counts['totals'] = val_col_counts.sum(axis = 1)
val_col_counts = val_col_counts.append(pd.Series(val_col_counts.sum(axis=0), name = 'subtotals'))
val_col_counts

df.to_csv("indeed_scrapy/scrape_data/company_reviews100to500fort20171020" + "num_n_cowide sorted" + ".csv")

