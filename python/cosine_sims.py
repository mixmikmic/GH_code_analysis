import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

pkl_file = open('../data/df.pkl', 'rb')
df = pickle.load(pkl_file)
pkl_file.close() 

features = [
            'top_elev_(ft)', 
            'bottom_elev_(ft)', 
            'vert_rise_(ft)', 
            'slope_length_(ft)', 
            'avg_width_(ft)', 
            'slope_area_(acres)', 
            'avg_grade_(%)', 
            'max_grade_(%)', 
            'groomed']

X = df[features].values

X

def cos_sim_recommendations(trail_name, resort_name, X, n=5, resort=None):
    index = df.index[(df['trail_name'] == trail_name) & (df['resort'] == resort_name)][0]
    trail = X[index].reshape(1,-1)
    cs = cosine_similarity(trail, X)
    rec_index = np.argsort(cs)[0][::-1][1:]
    ordered_df = df.loc[rec_index]
    if resort:
        ordered_df = ordered_df[ordered_df['resort'] == resort]
    rec_df = ordered_df.head(n)
    orig_row = df.loc[[index]].rename(lambda x: 'original')
    total = pd.concat((orig_row,rec_df))
    return total

ss = StandardScaler()
X = ss.fit_transform(X)

X

cos_sim_recommendations('Sorensen Park','Winter Park',X,n=5)

'''least similar'''
cs = cosine_similarity(X[0].reshape(1,-1), X)[0]
css = list(enumerate(cs))
srtd = sorted(css, key=lambda x: x[1])[::-1]
srtd[-5:]

df.iloc[[0,673,821,848,314,403]]

df.describe()

color = ['green','blue']

def cos_sim_recs(index, n=5, resort=None, color=None):
    trail = X[index].reshape(1,-1)
    cs = cosine_similarity(trail, X)
    rec_index = np.argsort(cs)[0][::-1][1:]
    ordered_df = df.loc[rec_index]
    if resort:
        ordered_df = ordered_df[ordered_df['resort'] == resort]
    if color:
        ordered_df = ordered_df[ordered_df['colors'].isin(color)]
    rec_df = ordered_df.head(n)
    rec_df = rec_df.reset_index(drop=True)
    rec_df.index = rec_df.index+1
    orig_row = df.loc[[index]].rename(lambda x: 'original')
    total = pd.concat((orig_row,rec_df))
    return total

cos_sim_recs(901,n=10,color=color)



results_df = cos_sim_recommendations('Sorensen Park','Winter Park',X,n=5)

results_df

dfs = []
for num in range(1,6):
    dfs.append(results_df.iloc[[0,num]].select_dtypes(include=[np.number]))
pct_changes = []
for num in range(0,5):
    pct_changes.append(dfs[num].pct_change().iloc[[1]])
pct_change_df = pd.concat(pct_changes)
pct_change_df

pct_change_df['length'] = 'Longer'
pct_change_df['length'][pct_change_df['slope_length_(ft)'] == 0] = 'Same Length'
pct_change_df['length'][pct_change_df['slope_length_(ft)'] < 0] = 'Shorter'

pct_change_df['avg_slope'] = 'Steeper Overall'
pct_change_df['avg_slope'][pct_change_df['avg_grade_(%)'] == 0] = 'Same Average Slope'
pct_change_df['avg_slope'][pct_change_df['avg_grade_(%)'] < 0] = 'Less Steep Overall'

pct_change_df['max_slope'] = 'Steeper Max slope'
pct_change_df['max_slope'][pct_change_df['max_grade_(%)'] == 0] = 'Same Max Slope'
pct_change_df['max_slope'][pct_change_df['max_grade_(%)'] < 0] = 'Less Steep Max Slope'

pct_change_df['vert_rise'] = 'More Vertical'
pct_change_df['vert_rise'][pct_change_df['vert_rise_(ft)'] == 0] = 'Same Vertical'
pct_change_df['vert_rise'][pct_change_df['vert_rise_(ft)'] < 0] = 'Less Vertical'

pct_change_df['width'] = 'Wider Overall'
pct_change_df['width'][pct_change_df['avg_width_(ft)'] == 0] = 'Same Width Overall'
pct_change_df['width'][pct_change_df['avg_width_(ft)'] < 0] = 'Narrower Overall'

pct_change_df



X

X[0]

X_weighted_pt1 = X[:,:-3]
X_weighted_pt2 = X[:,-3]*2  # avg_grade
X_weighted_pt3 = X[:,-2]*2  # max_grade
X_weighted_pt4 = X[:,-1]
X_weighted = np.hstack((X_weighted_pt1,X_weighted_pt2.reshape(-1,1),X_weighted_pt3.reshape(-1,1),X_weighted_pt4.reshape(-1,1),))

X_weighted_pt1.shape

X_weighted = np.hstack((X_weighted_pt1,X_weighted_pt2.reshape(-1,1),X_weighted_pt3.reshape(-1,1),X_weighted_pt4.reshape(-1,1),))

X_weighted

w_results_df = cos_sim_recommendations('Jack Kendrick','Winter Park',X_weighted,n=10,resort='Winter Park')

w_results_df

results_df = cos_sim_recommendations('Jack Kendrick','Winter Park',X,n=10,resort='Winter Park')

results_df



