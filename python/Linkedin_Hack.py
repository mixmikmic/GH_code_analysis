import numpy as np
import pandas as pd

applicant_data = pd.read_csv('applicants.csv')
team_data = pd.read_csv('team.csv')

team_url = team_data.User_url;
del team_data['User_url']

applicant_url = applicant_data['User_url']
del applicant_data['User_url']

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=1, random_state=0).fit(team_data)
kmeans.labels_

distance_data = kmeans.transform(applicant_data)

distance_df = pd.DataFrame(distance_data)
distance_df['User_url'] = applicant_url
sorted_distance_df = distance_df.sort_values(by=0, ascending=1)
sorted_distance = sorted_distance_df.reset_index()
del sorted_distance['index']
sorted_distance

required_data = sorted_distance[:20]
#required_data
applicant_data['User_url'] = applicant_url

#new_data= required_data.join(applicant_data, on='User_url',how='inner',rsuffix = '_User_url')
result = pd.merge(required_data, applicant_data, how='inner', on='User_url')
result

new_result = result.transpose()
new_result
#new_result.to_csv('shortlisted_candidates.csv')

#r = result.mean().to_frame(name='sum')
#r
r = team_data.mean().to_frame(name='avg')
r

r= r.reset_index()
r[2:]

new_r = r[2:]
new_r

new_r.to_csv("shortlisted_candidates.csv", index=False)

final_result = pd.concat([r, new_result], axis=1, join='inner')
final_result.to_csv('shortlisted_candidates.csv')



