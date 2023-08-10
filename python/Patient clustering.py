from IPython.core.display import HTML
css_file = 'style.css'
HTML(open(css_file, 'r').read())

from pandas import read_excel, merge

from numpy import arange

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Plotly requires pip install plotly
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()

# Importing an Excel spreadsheet with two sheets as two DataFrames
df_campaign = read_excel("PatientResponse.xlsx", sheetname = 0)

df_response = read_excel("PatientResponse.xlsx", sheetname = 1)
# Adding a column of value 1 to act as a count for that instance
df_response["n"] = 1

df_campaign.tail()

df_response.tail()

# Merge on the CampaignID columns
df = merge(df_campaign, df_response, on = "CampaignID")

df.tail()

# Create a pivot table to count each of the 32 campaigns
table = df.pivot_table(index = ["Patient"], columns = ["CampaignID"], values = "n")

table.tail()

# Fill NA values with 0 and reset the index to CampaignID
table = table.fillna(0).reset_index()

table.tail()

# Extracting the columns (32 campaigns)
cols = table.columns[1:]

cols

cluster = KMeans(n_clusters = 5) # At least 7-times times cluster = patients

# Predict the cluster from first patient down all the rows
table["cluster"] = cluster.fit_predict(table[table.columns[2:]])

table.tail()

# Principal component separation to create a 2-dimensional picture
pca = PCA(n_components = 2)
table['x'] = pca.fit_transform(table[cols])[:,0]
table['y'] = pca.fit_transform(table[cols])[:,1]
table = table.reset_index()

table.tail()

patient_clusters = table[["Patient", "cluster", "x", "y"]]

patient_clusters.tail()

final = merge(df_response, patient_clusters)
final = merge(df_campaign, final);

final.tail()

trace0 = go.Scatter(x = patient_clusters[patient_clusters.cluster == 0]["x"],
                    y = patient_clusters[patient_clusters.cluster == 0]["y"],
                    name = "Cluster 1",
                    mode = "markers",
                    marker = dict(size = 10,
                                 color = "rgba(15, 152, 152, 0.5)",
                                 line = dict(width = 1, color = "rgb(0,0,0)")))
trace1 = go.Scatter(x = patient_clusters[patient_clusters.cluster == 1]["x"],
                    y = patient_clusters[patient_clusters.cluster == 1]["y"],
                    name = "Cluster 2",
                    mode = "markers",
                    marker = dict(size = 10,
                                 color = "rgba(180, 18, 180, 0.5)",
                                 line = dict(width = 1, color = "rgb(0,0,0)")))
trace2 = go.Scatter(x = patient_clusters[patient_clusters.cluster == 2]["x"],
                    y = patient_clusters[patient_clusters.cluster == 2]["y"],
                    name = "Cluster 3",
                    mode = "markers",
                    marker = dict(size = 10,
                                 color = "rgba(132, 132, 132, 0.8)",
                                 line = dict(width = 1, color = "rgb(0,0,0)")))
trace3 = go.Scatter(x = patient_clusters[patient_clusters.cluster == 3]["x"],
                    y = patient_clusters[patient_clusters.cluster == 3]["y"],
                    name = "Cluster 4",
                    mode = "markers",
                    marker = dict(size = 10,
                                 color = "rgba(122, 122, 12, 0.8)",
                                 line = dict(width = 1, color = "rgb(0,0,0)")))
trace4 = go.Scatter(x = patient_clusters[patient_clusters.cluster == 4]["x"],
                    y = patient_clusters[patient_clusters.cluster == 4]["y"],
                    name = "Cluster 5",
                    mode = "markers",
                    marker = dict(size = 10,
                                 color = "rgba(230, 20, 30, 0.5)",
                                 line = dict(width = 1, color = "rgb(0,0,0)")))

data = [trace0, trace1, trace2, trace3, trace4]

iplot(data)

# e-mails, short-message services, WhatsApp messages, pamphlets, telephone and long letters
final["0"] = final.cluster == 0
final.groupby("0").Type.value_counts()

# Number of patients in this cluster
final[final.cluster == 0]["Patient"].count()

# List of patients
final[final.cluster == 0]["Patient"]



