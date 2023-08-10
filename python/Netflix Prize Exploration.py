import pandas as pd
pd.set_option('display.float_format', lambda x: '%.3f' % x)
import numpy as np
import math
import re
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import Reader, Dataset, SVD, evaluate

combined_data_1_df = pd.read_csv("../netflix-prize-data/combined_data_1.txt", 
                          header = None, 
                          names = ['Cust_Id', 'Rating'], usecols = [0,1])
combined_data_1_df["Rating"] = combined_data_1_df["Rating"].astype(float)

combined_data_1_df.head()

df_title = pd.read_csv('../netflix-prize-data/movie_titles.csv', encoding = "ISO-8859-1", header = None, names = ['Movie_Id', 'Year', 'Name'])
df_title.set_index('Movie_Id', inplace = True)
print (df_title.head())



