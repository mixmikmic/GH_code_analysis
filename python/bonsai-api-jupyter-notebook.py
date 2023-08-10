get_ipython().system('pip install pandas seaborn matplotlib requests plotly python-dateutil cufflinks')

import pandas as pd # for Dataframes
import seaborn as sns # for matplotlib theme
import matplotlib.pyplot as plt
import requests, plotly, dateutil
import cufflinks as cf # ties plotly and dataframes
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
plotly.offline.init_notebook_mode(connected=True)
cf.go_offline()

# setting seaborn theme for matplotlib
sns.set()
sns.set_context("talk")
# to plot figures in the notebook
get_ipython().magic('matplotlib inline')
# for high resolution plots hd displays (macbook pros)
get_ipython().magic("config InlineBackend.figure_format = 'retina'")

# FILL IN BELOW
username = 'YourUserName'
AccessKey = 'YourAccessKey'

rootUrl = 'https://api.bons.ai' # this is the API URL for beta.bons.ai

endpoint = '/v1/' + username
url = rootUrl + endpoint # define url
print("The URL we are listing all BRAINs for: " + url)

r = requests.get(url,headers={'Authorization': AccessKey})
data = r.json() # retrieve data from json format into a dictionary
datadf = pd.DataFrame(data['brains']) # import date from dict to Dataframe for ease of data analysis
datadf # show dataframe

# FILL IN BELOW
brainname = 'cartpole-jupyter' # use a BRAIN name listed in the previous table

endpoint = '/v1/'+ username +'/'+ brainname
url = rootUrl + endpoint
print("The URL for the specified BRAIN: " + url)

r = requests.get(url,headers={'Authorization':AccessKey})
data = r.json()
datadf = pd.DataFrame(data['versions'])
datadf # show dataframe

# FILL IN BELOW
brainversion = 'latest' # can be latest version 'latest' or a number e.g. '3' for version 3

metrics = '/metrics/episode_value' 
endpoint = '/v1/' + username + '/' + brainname + '/' + brainversion + metrics
url = rootUrl + endpoint
print("The URL for the specified BRAIN version: " + url)

r = requests.get(url,headers={'Authorization': AccessKey})
data = r.json()
datadf = pd.DataFrame(data)
datadf.tail() # tail displays only the last 5 lines of the output

datadf['value'].iplot(
    xTitle='episode index',
    yTitle='Episode reward',
    title='Episode reward vs episode index',
    kind='scatter',
    mode='markers',
    size=5,
)

mean_reward = datadf['value'].rolling(100,center = True).mean().values # compute rolling mean
std_reward = datadf['value'].rolling(100,center = True).std().values # compute rolling standard deviation
episode_index = datadf['value'].index

# plot it!
fig, ax = plt.subplots(1,figsize=(20,10))
plt.figure(figsize=(3,4))
ax.plot(episode_index,mean_reward)# , label='mean population 1')
ax.fill_between(episode_index, mean_reward+std_reward, mean_reward-std_reward, alpha=0.5)
ax.set_title('mean episode reward TRAINING')
#ax.legend(loc='upper left')
ax.set_xlabel('episode index')
ax.set_ylabel('mean episode reward')

# FILL IN BELOW
brainversion = 'latest' # can be latest version 'latest' or a number e.g. '3' for version 3

metrics = '/metrics/test_pass_value' 
endpoint = '/v1/' + username + '/' + brainname + '/' + brainversion + metrics
url = rootUrl + endpoint
print("The URL for the specified BRAIN version: " + url)

r = requests.get(url,headers={'Authorization': AccessKey})
data = r.json()
datadf = pd.DataFrame(data)
datadf.tail() # tail displays only the last 5 lines of the output

datadf[['iteration','value']].iplot(
    x='iteration',
    y='value',
    xTitle='Iteration index',
    yTitle='Episode reward',
    title='Test episode data',
    kind='scatter',
    mode='markers',
    size=5,
)

mean_reward = datadf['value'].rolling(10).mean().values # compute rolling mean
iteration_index = datadf['iteration']

# plot it!
fig, ax = plt.subplots(1,figsize=(20,10))
plt.figure(figsize=(3,4))
ax.plot(iteration_index,mean_reward)# , label='mean population 1') we're plotting 
ax.scatter(iteration_index,datadf['value'],alpha=0.7)
ax.set_title('Mean episode test pass with data')
ax.set_xlabel('Iteration index')
ax.set_ylabel('Episode reward')



