import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
import time

pkl_file = open('../data/mtn_df.pkl', 'rb')
mtn_df = pickle.load(pkl_file)
pkl_file.close() 

mtn_df.head()

features = ['top_elev_(ft)', 
            'bottom_elev_(ft)', 
            'vert_rise_(ft)', 
            'slope_length_(ft)', 
            'avg_width_(ft)', 
            'slope_area_(acres)', 
            'avg_grade_(%)', 
            'max_grade_(%)', 
            'groomed',
            'resort_bottom',
            'resort_top',
            'greens',
            'blues',
            'blacks',
            'bbs',
            'lifts',
            'price']

X = mtn_df[features].values

X

def mtn_recommendations(trail_name, resort_name, X, n=5):
    index = mtn_df.index[(mtn_df['trail_name'] == trail_name) & (mtn_df['resort'] == resort_name)][0]
    trail = X[index].reshape(1,-1)
    cs = cosine_similarity(trail, X)[0]
    mtn_df['cosine_sim'] = cs
    s = mtn_df.groupby('resort').mean()['cosine_sim'].sort_values()[::-1]
    orig_row = mtn_df.loc[[index]].rename(lambda x: 'original')
    return list(s.index[:n])

ss = StandardScaler()
X = ss.fit_transform(X)

recs = mtn_recommendations('So Fine','Copper',X,n=5)
recs

cs = cosine_similarity(X[300].reshape(1,-1), X)[0]
cs

mtn_df['cosine_sim'] = cs

mtn_df.groupby('resort').mean()['cosine_sim'].sort_values()[::-1]

_.index[0]

mtn_df.iloc[300]

resort_stats_df = mtn_df[['resort', 'resort_bottom','resort_top','greens','blues','blacks','bbs','lifts','price']].drop_duplicates()

resort_stats_df

results_df = pd.DataFrame(columns=['resort', 'resort_bottom','resort_top','greens','blues','blacks','bbs','lifts','price'])
for rec in recs:
    results_df = results_df.append(resort_stats_df[resort_stats_df['resort'] == rec])

results_df

def clean_df_for_recs(df):
    df['groomed'][df['groomed'] == 1] = 'Groomed'
    df['groomed'][df['groomed'] == 0] = 'Ungroomed'
    df['color_names'] = df['colors']
    df['color_names'][df['color_names'] == 'green'] = 'Green'
    df['color_names'][df['color_names'] == 'blue'] = 'Blue'
    df['color_names'][df['color_names'] == 'black'] = 'Black'
    df['color_names'][df['color_names'] == 'bb'] = 'Double Black'
    df = df[['trail_name','resort','location','color_names','groomed','top_elev_(ft)','bottom_elev_(ft)','vert_rise_(ft)','slope_length_(ft)','avg_width_(ft)','slope_area_(acres)','avg_grade_(%)','max_grade_(%)']]
    df.columns = ['Trail Name', 'Resort','Location','Difficulty','Groomed','Top Elev (ft)', 'Bottom Elev (ft)', 'Vert Rise (ft)', 'Slope Length (ft)', 'Avg Width (ft)', 'Slope Area (acres)', 'Avg Grade (%)', 'Max Grade (%)']
    return df

clean_df_for_recs(mtn_df)

results_df

results_df.colummns = ['Resort','Resort Bottom Elevation', 'Resort Top Elevations', 'Percent Greens', 'Percent Blues', 'Percent Blacks', 'Percent Double  Blacks', 'Number of Lifts', 'Price']

for i,row in results_df.iterrows():
    print(row.resort)

recs = mtn_recommendations('So Fine','Copper',X,n=5)
recs



