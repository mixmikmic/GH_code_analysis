import pandas as pd
import numpy as np
df = pd.read_csv('data/all_results.csv')

non_empty_countries = df['country'].dropna()
non_empty_countries.groupby(non_empty_countries.values).count().sort_values(ascending=False)

city_series = df['city'].dropna().drop_duplicates(keep='first')
city_series

import re

def get_num_words(name):
    name_words = re.split(r'[\s-]+', name)
    return len(name_words)
    
city_words_count_series = pd.Series(city_series.map(get_num_words).values, city_series.values)
city_words_count_series.groupby(city_words_count_series.values).count()

city_words_count_series[city_words_count_series > 2]



