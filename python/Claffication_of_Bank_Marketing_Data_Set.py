#Packages
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns 
from pylab import plot, show
get_ipython().run_line_magic('matplotlib', 'inline')

#current directory
get_ipython().system('pwd')

#Reading the csv files
bank_additional_full_df=pd.read_csv('../data/bank-additional/bank-additional-full.csv',sep=';')
bank_additional_df=pd.read_csv('../data/bank-additional/bank-additional.csv',sep=';')
bank_full_df=pd.read_csv('../data/bank/bank-full.csv',sep=';')
bank_df=pd.read_csv('../data/bank/bank.csv',sep=';')

#Columns information
bank_additional_full_df.columns

#size
bank_additional_full_df.shape

#Info
bank_additional_full_df.info()

#Describe
bank_additional_full_df.describe()

#Getting first five rows
bank_additional_full_df.head()

#Checking for null value
bank_additional_full_df.isnull().sum()

#plotting employment variation rate - quarterly indicator emp.var.rate
plt.rcParams['figure.figsize'] = (8, 6)
sns.countplot(x='emp.var.rate', hue='emp.var.rate', data=bank_additional_full_df);

#previous: number of contacts performed before this campaign and for this client (numeric)
plt.rcParams['figure.figsize'] = (8, 6)
sns.countplot(x='previous',hue='previous',data=bank_additional_full_df);

#plot data
fig, ax = plt.subplots(figsize=(15,7))
bank_additional_full_df.groupby(['duration']).count()[['education','nr.employed']].plot(ax=ax)

# Another way to plot a histogram of duration is shown below
bank_additional_full_df['duration'].hist(bins=50)

#Describing dummy keys of particular column
y_n_lookup ={'yes' : 1, 'no' : 0}
bank_additional_full_df['y_dummy'] = bank_additional_full_df['y'].map(lambda x: y_n_lookup[x])
bank_additional_full_df['y_dummy'].value_counts()

#getting marital status of groupby people
age_group_names = ['young', 'lower middle', 'middle', 'senior']
bank_additional_full_df['age_binned'] = pd.qcut(bank_additional_full_df['age'], 4, labels = age_group_names)
bank_additional_full_df['age_binned'].value_counts()
gb_marital_age = bank_additional_full_df['y_dummy'].groupby([bank_additional_full_df['marital'],bank_additional_full_df['age_binned']] ) 
gb_marital_age.mean()

#unstack (Pivot a level of the (necessarily hierarchical) index labels) groupby marital status
gb_marital_age.mean().unstack()

#getting life stage of age group
bank_additional_full_df['life_stage'] = bank_additional_full_df.apply(lambda x: x['age_binned'] +' & ' + x['marital'], axis = 1)
bank_additional_full_df['life_stage'].value_counts() 

from sklearn import preprocessing
from sklearn import cluster
import matplotlib.pyplot as plt
#getting the pattern of particular age range employee
combined_data = bank_additional_full_df[['age','nr.employed']].as_matrix()
combined_data_scaled = preprocessing.scale(combined_data)

# Applying KMeans algorithm
kmeans = cluster.KMeans(n_clusters = 3)
kmeans.fit(combined_data_scaled)
y_pred = kmeans.predict(combined_data_scaled)
#Plotting the graph
plt.scatter(combined_data_scaled[:, 0], combined_data_scaled[:, 1], c = y_pred)
plt.xlabel('Scaled Age')
plt.ylabel('Scaled  number of employees')
plt.show()

#Consider some important features
feature_col=['job', 'marital', 'education', 'default', 'housing', 'loan',
             'contact', 'month', 'day_of_week','previous', 'poutcome']

#importing the packeges
from IPython.core.interactiveshell import InteractiveShell     #An enhanced, interactive shell for Python
#‘all’, ‘last’, ‘last_expr’ or ‘none’, ‘last_expr_or_assign’
#specifying which nodes should be run interactively 
import plotly 
plotly.tools.set_credentials_file(username='KunalBhashkar', api_key='3ImJpD57ThNbPx117FsM')
InteractiveShell.ast_node_interactivity = "all"   #Options:	'all','last','last_expr','none','last_expr_or_assign'

import numpy as np # linear algebra
import pandas as pd # data processing

#Plotly Offline brings interactive Plotly graphs to the offline (local) environment
import plotly.offline as py
import plotly.figure_factory as ff
import plotly.graph_objs as gobj
import plotly.plotly as plty
from plotly.graph_objs import *
py.init_notebook_mode(connected=True)  # initiate notebook for offline plot

#Method for plotting 

def plot_value_counts(col_name,table=False,bar=False):
    
    values_count = pd.DataFrame(bank_additional_full_df[col_name].value_counts())
    values_count.columns = ['count']
   
    # Converting the index column into value count
    values_count[col_name] = [ str(i) for i in values_count.index ]
    
    # add a column with the percentage of each data point to the sum of all data points
    values_count['percent'] = values_count['count'].div(values_count['count'].sum()).multiply(100).round(2)
    
    # change the order of the columns.
    values_count = values_count.reindex([col_name,'count','percent'],axis=1)
    values_count.reset_index(drop=True,inplace=True)    
    
    if bar :
        # add a font size for annotations0 which is relevant to the length of the data points.
        font_size = 20 - (.25 * len(values_count[col_name]))
        
        trace0 = gobj.Bar( x = values_count[col_name], y = values_count['count'] )
        data_ = gobj.Data( [trace0] )
        
        annotations0 = [ dict(x = xi,
                             y = yi, 
                             showarrow=False,
                             font={'size':font_size},
                             text = "{:,}".format(yi),
                             xanchor='center',
                             yanchor='bottom' )
                       for xi,yi,_ in values_count.values ]
        
        annotations1 = [ dict( x = xi,
                              y = yi/2,
                              showarrow = False,
                              text = "{}%".format(pi),
                              xanchor = 'center',
                              yanchor = 'center',
                              font = {'color':'yellow'})
                         for xi,yi,pi in values_count.values if pi > 10 ]
        
        annotations = annotations0 + annotations1                       
        
        layout = gobj.Layout( title = col_name.replace('_',' ').capitalize(),
                             titlefont = {'size': 50},
                             yaxis = {'title':'count'},
                             xaxis = {'type':'category'},
                            annotations = annotations  )
        figure = gobj.Figure( data = data_, layout = layout )
        py.iplot(figure)
    
    if table : 
        values_count['count'] = values_count['count'].apply(lambda d : "{:,}".format(d))
        table = ff.create_table(values_count,index_title="race")      #Creating the table for race
        py.iplot(table)
    
    return values_count

for col in feature_col:
    _ = plot_value_counts(col,0,1)

#Columns information
bank_additional_df.columns

#size
bank_additional_df.shape

#Info
bank_additional_df.info()

#Describe
bank_additional_df.describe()

#Getting first five rows
bank_additional_df.head()

#Checking for null value
bank_additional_df.isnull().sum()

#plotting employment variation rate - quarterly indicator emp.var.rate
plt.rcParams['figure.figsize'] = (8, 6)
sns.countplot(x='emp.var.rate', hue='emp.var.rate', data=bank_additional_df);

#previous: number of contacts performed before this campaign and for this client (numeric)
plt.rcParams['figure.figsize'] = (8, 6)
sns.countplot(x='previous',hue='previous',data=bank_additional_df);

#plot data
fig, ax = plt.subplots(figsize=(15,7))
bank_additional_df.groupby(['duration']).count()[['education','nr.employed']].plot(ax=ax)

# Another way to plot a histogram of duration is shown below
bank_additional_df['duration'].hist(bins=50)

#Describing dummy keys of particular column
y_n_lookup ={'yes' : 1, 'no' : 0}
bank_additional_df['y_dummy'] = bank_additional_df['y'].map(lambda x: y_n_lookup[x])
bank_additional_df['y_dummy'].value_counts()

#getting marital status of groupby people
age_group_names = ['young', 'lower middle', 'middle', 'senior']
bank_additional_df['age_binned'] = pd.qcut(bank_additional_df['age'], 4, labels = age_group_names)
gb_marital_age = bank_additional_df['y_dummy'].groupby([bank_additional_df['marital'],bank_additional_df['age_binned']] ) 
gb_marital_age.value_counts()

#unstack (Pivot a level of the (necessarily hierarchical) index labels) groupby marital status
gb_marital_age.mean().unstack()

#getting life stage of age group
bank_additional_df['life_stage'] = bank_additional_df.apply(lambda x: x['age_binned'] +' & ' + x['marital'], axis = 1)
bank_additional_df['life_stage'].value_counts()

from sklearn import preprocessing
from sklearn import cluster
import matplotlib.pyplot as plt
#getting the pattern of particular age range employee
combined_data = bank_additional_df[['age','nr.employed']].as_matrix()
combined_data_scaled = preprocessing.scale(combined_data)

# Applying KMeans algorithm
kmeans = cluster.KMeans(n_clusters = 3)
kmeans.fit(combined_data_scaled)
y_pred = kmeans.predict(combined_data_scaled)
#Plotting the graph
plt.scatter(combined_data_scaled[:, 0], combined_data_scaled[:, 1], c = y_pred)
plt.xlabel('Scaled Age')
plt.ylabel('Scaled  number of employees')
plt.show()

#Consider some important features
feature_col=['job', 'marital', 'education', 'default', 'housing', 'loan',
             'contact', 'month', 'day_of_week','previous', 'poutcome']

#importing the packeges
from IPython.core.interactiveshell import InteractiveShell     #An enhanced, interactive shell for Python
#‘all’, ‘last’, ‘last_expr’ or ‘none’, ‘last_expr_or_assign’
#specifying which nodes should be run interactively 
import plotly 
plotly.tools.set_credentials_file(username='KunalBhashkar', api_key='3ImJpD57ThNbPx117FsM')
InteractiveShell.ast_node_interactivity = "all"   #Options:	'all','last','last_expr','none','last_expr_or_assign'

import numpy as np # linear algebra
import pandas as pd # data processing

#Plotly Offline brings interactive Plotly graphs to the offline (local) environment
import plotly.offline as py
import plotly.figure_factory as ff
import plotly.graph_objs as gobj
import plotly.plotly as plty
from plotly.graph_objs import *
py.init_notebook_mode(connected=True)  # initiate notebook for offline plot

#Method for plotting 

def plot_value_counts(col_name,table=False,bar=False):
    
    values_count = pd.DataFrame(bank_additional_df[col_name].value_counts())
    values_count.columns = ['count']
   
    # Converting the index column into value count
    values_count[col_name] = [ str(i) for i in values_count.index ]
    
    # add a column with the percentage of each data point to the sum of all data points
    values_count['percent'] = values_count['count'].div(values_count['count'].sum()).multiply(100).round(2)
    
    # change the order of the columns.
    values_count = values_count.reindex([col_name,'count','percent'],axis=1)
    values_count.reset_index(drop=True,inplace=True)    
    
    if bar :
        # add a font size for annotations0 which is relevant to the length of the data points.
        font_size = 20 - (.25 * len(values_count[col_name]))
        
        trace0 = gobj.Bar( x = values_count[col_name], y = values_count['count'] )
        data_ = gobj.Data( [trace0] )
        
        annotations0 = [ dict(x = xi,
                             y = yi, 
                             showarrow=False,
                             font={'size':font_size},
                             text = "{:,}".format(yi),
                             xanchor='center',
                             yanchor='bottom' )
                       for xi,yi,_ in values_count.values ]
        
        annotations1 = [ dict( x = xi,
                              y = yi/2,
                              showarrow = False,
                              text = "{}%".format(pi),
                              xanchor = 'center',
                              yanchor = 'center',
                              font = {'color':'yellow'})
                         for xi,yi,pi in values_count.values if pi > 10 ]
        
        annotations = annotations0 + annotations1                       
        
        layout = gobj.Layout( title = col_name.replace('_',' ').capitalize(),
                             titlefont = {'size': 50},
                             yaxis = {'title':'count'},
                             xaxis = {'type':'category'},
                            annotations = annotations  )
        figure = gobj.Figure( data = data_, layout = layout)
        py.iplot(figure)
    
    if table : 
        values_count['count'] = values_count['count'].apply(lambda d : "{:,}".format(d))
        table = ff.create_table(values_count,index_title="race")      #Creating the table for race
        py.iplot(table)
    
    return values_count

for col in feature_col:
    _ = plot_value_counts(col,0,1)

#Columns information
bank_full_df.columns

#size
bank_full_df.shape

#Info
bank_full_df.info()

#Describe
bank_full_df.describe()

#Getting first five rows
bank_full_df.head()

#Checking for null value
bank_full_df.isnull().sum()

#plot data
fig, ax = plt.subplots(figsize=(15,7))
bank_full_df.groupby(['duration']).count()[['job','education']].plot(ax=ax)

# Another way to plot a histogram of duration is shown below
bank_full_df['duration'].hist(bins=50)

#Describing dummy keys of particular column
y_n_lookup ={'yes' : 1, 'no' : 0}
bank_full_df['y_dummy'] = bank_full_df['y'].map(lambda x: y_n_lookup[x])
bank_full_df['y_dummy'].value_counts()

#getting marital status of groupby people
age_group_names = ['young', 'lower middle', 'middle', 'senior']
bank_full_df['age_binned'] = pd.qcut(bank_full_df['age'], 4, labels = age_group_names)
gb_marital_age = bank_full_df['y_dummy'].groupby([bank_full_df['marital'],bank_full_df['age_binned']] ) 
gb_marital_age.value_counts()

#unstack (Pivot a level of the (necessarily hierarchical) index labels) groupby marital status
gb_marital_age.mean().unstack()

#getting life stage of age group
bank_full_df['life_stage'] = bank_full_df.apply(lambda x: x['age_binned'] +' & ' + x['marital'], axis = 1)
bank_full_df['life_stage'].value_counts() 

#Scale the data of age and balance
from sklearn import preprocessing
from sklearn import cluster
combined_data = bank_full_df[['age','balance']].as_matrix()
combined_data_scaled = preprocessing.scale(combined_data)
#Applying KMeans clustering for prediction
kmeans = cluster.KMeans(n_clusters = 3)
kmeans.fit(combined_data_scaled)
y_pred = kmeans.predict(combined_data_scaled)
#Plotting the age and balance
plt.scatter(combined_data_scaled[:, 0], combined_data_scaled[:, 1], c = y_pred)
plt.xlabel('Scaled Age')
plt.ylabel('Scaled Balance')
plt.show()

#Consider some important features
feature_col=['job', 'marital', 'education','default','balance','housing', 'loan',
             'contact', 'month','month','previous','poutcome']

#importing the packeges
from IPython.core.interactiveshell import InteractiveShell     #An enhanced, interactive shell for Python
#‘all’, ‘last’, ‘last_expr’ or ‘none’, ‘last_expr_or_assign’
#specifying which nodes should be run interactively 
import plotly 
plotly.tools.set_credentials_file(username='KunalBhashkar', api_key='3ImJpD57ThNbPx117FsM')
InteractiveShell.ast_node_interactivity = "all"   #Options:	'all','last','last_expr','none','last_expr_or_assign'

import numpy as np # linear algebra
import pandas as pd # data processing

#Plotly Offline brings interactive Plotly graphs to the offline (local) environment
import plotly.offline as py
import plotly.figure_factory as ff
import plotly.graph_objs as gobj
import plotly.plotly as plty
from plotly.graph_objs import *
py.init_notebook_mode(connected=True)  # initiate notebook for offline plot

#Method for plotting 

def plot_value_counts(col_name,table=False,bar=False):
    
    values_count = pd.DataFrame(bank_full_df[col_name].value_counts())
    values_count.columns = ['count']
   
    # Converting the index column into value count
    values_count[col_name] = [ str(i) for i in values_count.index ]
    
    # add a column with the percentage of each data point to the sum of all data points
    values_count['percent'] = values_count['count'].div(values_count['count'].sum()).multiply(100).round(2)
    
    # change the order of the columns.
    values_count = values_count.reindex([col_name,'count','percent'],axis=1)
    values_count.reset_index(drop=True,inplace=True)    
    
    if bar :
        # add a font size for annotations0 which is relevant to the length of the data points.
        font_size = 20 - (.25 * len(values_count[col_name]))
        
        trace0 = gobj.Bar( x = values_count[col_name], y = values_count['count'] )
        data_ = gobj.Data( [trace0] )
        
        annotations0 = [ dict(x = xi,
                             y = yi, 
                             showarrow=False,
                             font={'size':font_size},
                             text = "{:,}".format(yi),
                             xanchor='center',
                             yanchor='bottom' )
                       for xi,yi,_ in values_count.values ]
        
        annotations1 = [ dict( x = xi,
                              y = yi/2,
                              showarrow = False,
                              text = "{}%".format(pi),
                              xanchor = 'center',
                              yanchor = 'center',
                              font = {'color':'yellow'})
                         for xi,yi,pi in values_count.values if pi > 10 ]
        
        annotations = annotations0 + annotations1                       
        
        layout = gobj.Layout( title = col_name.replace('_',' ').capitalize(),
                             titlefont = {'size': 50},
                             yaxis = {'title':'count'},
                             xaxis = {'type':'category'},
                            annotations = annotations  )
        figure = gobj.Figure( data = data_, layout = layout )
        py.iplot(figure)
    
    if table : 
        values_count['count'] = values_count['count'].apply(lambda d : "{:,}".format(d))
        table = ff.create_table(values_count,index_title="race")      #Creating the table for race
        py.iplot(table)
    
    return values_count

for col in feature_col:
    _ = plot_value_counts(col,0,1)

#Categorize the column of dataset which is object type 
for col in bank_full_df.columns:
    if bank_full_df[col].dtype == object:
        bank_full_df[col] = bank_full_df[col].astype('category')

#Convert categorical data into numerical value
bank_full_df["education"] = bank_full_df["education"].cat.codes

#Convert categorical data into numerical value
bank_full_df["job"] = bank_full_df["job"].cat.codes

bank_full_df.head()

from sklearn import preprocessing
from sklearn import cluster
import matplotlib.pyplot as plt
#getting the pattern of particular age range employee
econ_data = bank_full_df[['job','education']].as_matrix()
econ_data_scaled = preprocessing.scale(econ_data)

# Applying KMeans algorithm
kmeans = cluster.KMeans(n_clusters = 3)
kmeans.fit(econ_data_scaled)
y_pred = kmeans.predict(econ_data_scaled)
#Plotting the graph
plt.scatter(econ_data_scaled[:, 0], econ_data_scaled[:, 1], c = y_pred)
plt.xlabel('Scaled Job')
plt.ylabel('Scaled  Education')
plt.show()

#Columns information
bank_df.columns

#size
bank_df.shape

#Info
bank_df.info()

#Describe
bank_df.describe()

#Getting first five rows
bank_df.head()

#Checking for null value
bank_df.isnull().sum()

#plot data
fig, ax = plt.subplots(figsize=(15,7))
bank_df.groupby(['duration']).count()[['education','job']].plot(ax=ax)

# Another way to plot a histogram of duration is shown below
bank_df['duration'].hist(bins=50)

#Describing dummy keys of particular column
y_n_lookup ={'yes' : 1, 'no' : 0}
bank_df['y_dummy'] = bank_df['y'].map(lambda x: y_n_lookup[x])
bank_df['y_dummy'].value_counts()

#getting marital status of groupby people
age_group_names = ['young', 'lower middle', 'middle', 'senior']
bank_df['age_binned'] = pd.qcut(bank_df['age'], 4, labels = age_group_names)
gb_marital_age = bank_df['y_dummy'].groupby([bank_df['marital'],bank_df['age_binned']] ) 
gb_marital_age.value_counts()

#unstack (Pivot a level of the (necessarily hierarchical) index labels) groupby marital status
gb_marital_age.mean().unstack()

#getting life stage of age group
bank_df['life_stage'] = bank_df.apply(lambda x: x['age_binned'] +' & ' + x['marital'], axis = 1)
bank_df['life_stage'].value_counts() 

#Categorize the column of dataset which is object type 
for col in bank_df.columns:
    if bank_df[col].dtype == object:
        bank_df[col] = bank_df[col].astype('category')

#Convert categorical data into numerical value
bank_df["education"] = bank_df["education"].cat.codes

#Convert categorical data into numerical value
bank_df["job"] = bank_df["job"].cat.codes

from sklearn import preprocessing
from sklearn import cluster
import matplotlib.pyplot as plt
#getting the pattern of particular age range employee
combined_data = bank_df[['job','education']].as_matrix()
combined_data_scaled = preprocessing.scale(combined_data)

# Applying KMeans algorithm
kmeans = cluster.KMeans(n_clusters = 3)
kmeans.fit(combined_data_scaled)
y_pred = kmeans.predict(combined_data_scaled)
#Plotting the graph
plt.scatter(combined_data_scaled[:, 0], combined_data_scaled[:, 1], c = y_pred)
plt.xlabel('Scaled Job')
plt.ylabel('Scaled  Education')
plt.show()

#Scale the data of age and balance
from sklearn import preprocessing
from sklearn import cluster
combined_data = bank_df[['age','balance']].as_matrix()
combined_data_scaled = preprocessing.scale(combined_data)
#Applying KMeans clustering for prediction
kmeans = cluster.KMeans(n_clusters = 3)
kmeans.fit(combined_data_scaled)
y_pred = kmeans.predict(combined_data_scaled)
#Plotting the age and balance
plt.scatter(combined_data_scaled[:, 0], combined_data_scaled[:, 1], c = y_pred)
plt.xlabel('Scaled Age')
plt.ylabel('Scaled Balance')
plt.show()

#Consider some important features
feature_col=['job', 'marital', 'education', 'default','balance','housing', 'loan',
             'contact', 'month', 'day','campaign', 'pdays','previous','poutcome']

#importing the packeges
from IPython.core.interactiveshell import InteractiveShell     #An enhanced, interactive shell for Python
#‘all’, ‘last’, ‘last_expr’ or ‘none’, ‘last_expr_or_assign’
#specifying which nodes should be run interactively 
import plotly 
plotly.tools.set_credentials_file(username='KunalBhashkar', api_key='3ImJpD57ThNbPx117FsM')
InteractiveShell.ast_node_interactivity = "all"   #Options:	'all','last','last_expr','none','last_expr_or_assign'

import numpy as np # linear algebra
import pandas as pd # data processing

#Plotly Offline brings interactive Plotly graphs to the offline (local) environment
import plotly.offline as py
import plotly.figure_factory as ff
import plotly.graph_objs as gobj
import plotly.plotly as plty
from plotly.graph_objs import *
py.init_notebook_mode(connected=True)  # initiate notebook for offline plot

#Method for plotting 

def plot_value_counts(col_name,table=False,bar=False):
    
    values_count = pd.DataFrame(bank_df[col_name].value_counts())
    values_count.columns = ['count']
   
    # Converting the index column into value count
    values_count[col_name] = [ str(i) for i in values_count.index ]
    
    # add a column with the percentage of each data point to the sum of all data points
    values_count['percent'] = values_count['count'].div(values_count['count'].sum()).multiply(100).round(2)
    
    # change the order of the columns.
    values_count = values_count.reindex([col_name,'count','percent'],axis=1)
    values_count.reset_index(drop=True,inplace=True)    
    
    if bar :
        # add a font size for annotations0 which is relevant to the length of the data points.
        font_size = 20 - (.25 * len(values_count[col_name]))
        
        trace0 = gobj.Bar( x = values_count[col_name], y = values_count['count'] )
        data_ = gobj.Data( [trace0] )
        
        annotations0 = [ dict(x = xi,
                             y = yi, 
                             showarrow=False,
                             font={'size':font_size},
                             text = "{:,}".format(yi),
                             xanchor='center',
                             yanchor='bottom' )
                       for xi,yi,_ in values_count.values ]
        
        annotations1 = [ dict( x = xi,
                              y = yi/2,
                              showarrow = False,
                              text = "{}%".format(pi),
                              xanchor = 'center',
                              yanchor = 'center',
                              font = {'color':'yellow'})
                         for xi,yi,pi in values_count.values if pi > 10 ]
        
        annotations = annotations0 + annotations1                       
        
        layout = gobj.Layout( title = col_name.replace('_',' ').capitalize(),
                             titlefont = {'size': 50},
                             yaxis = {'title':'count'},
                             xaxis = {'type':'category'},
                            annotations = annotations  )
        figure = gobj.Figure( data = data_, layout = layout )
        py.iplot(figure)
    
    if table : 
        values_count['count'] = values_count['count'].apply(lambda d : "{:,}".format(d))
        table = ff.create_table(values_count,index_title="race")      #Creating the table for race
        py.iplot(table)
    
    return values_count

for col in feature_col:
    _ = plot_value_counts(col,0,1)

#Columns
bank_additional_full_df.columns

#Taking only those column which will affect more to output
features_columns=['job','education', 'default', 'housing', 'loan',
           'month', 'day_of_week', 'duration', 'pdays',
       'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx',
       'cons.conf.idx', 'euribor3m','y']

bank_additional_full_df.info()

#Encode the categorical data
for col in bank_additional_full_df.columns:
    if bank_additional_full_df[col].dtype==object:
           bank_additional_full_df[col]=bank_additional_full_df[col].astype('category')
           bank_additional_full_df[col]=bank_additional_full_df[col].cat.codes

bank_additional_full_df[features_columns].head()

# Rescale data (between 0 and 1)
import pandas
import scipy
import numpy
from sklearn.preprocessing import MinMaxScaler
features_columns_df=bank_additional_full_df[features_columns]
array = features_columns_df.values
# separate array into input and output components
X = array[:,0:15]
Y = array[:,15]
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(X)
# summarize transformed data
numpy.set_printoptions(precision=3)
print(rescaledX[0:5,:])

# Random Forest Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
X = features_columns_df.iloc[:,0:15]
Y = features_columns_df.iloc[:,15]
# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
rf = RandomForestClassifier(random_state = 42)  # random_state is the seed used by the random number generator
#fitting the model
model = rf.fit(X_train, y_train)
# Find feature importance, print it
raw_feature_importance = model.feature_importances_.tolist()
feature_importance = [round(val * 100.0, 2) for val in raw_feature_importance]
print(zip(features_columns_df.columns, feature_importance))

#Getting the score of feature matrix and its target values 
model.score(X_test,y_test)

from sklearn.metrics import classification_report
# Model Prediction 
predictions = model.predict(X_test)
#Print the classification report
print(classification_report(y_true =y_test,y_pred = predictions))
np.sum(predictions)

#ROC curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

y_true=sorted(y_test)
y_score=sorted(predictions)
# Compute fpr, tpr, thresholds and roc auc
fpr, tpr, thresholds = roc_curve(y_true, y_score)
roc_auc = auc(y_true, y_score)

# Plot ROC curve
plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate or (1 - Specifity)')
plt.ylabel('True Positive Rate or (Sensitivity)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")

from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# split data into X and y
X = features_columns_df.iloc[:,0:15]
Y = features_columns_df.iloc[:,15]

# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

#ROC curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

y_true=sorted(y_test)
y_score=sorted(predictions)
# Compute fpr, tpr, thresholds and roc auc
fpr, tpr, thresholds = roc_curve(y_true, y_score)
roc_auc = auc(y_true, y_score)

# Plot ROC curve
plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate or (1 - Specifity)')
plt.ylabel('True Positive Rate or (Sensitivity)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# split data into X and y
X = features_columns_df.iloc[:,0:15]
Y = features_columns_df.iloc[:,15]
# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
mlp = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500)
mlp.fit(X_train,y_train)

from sklearn.metrics import classification_report,confusion_matrix
predictions = mlp.predict(X_test)
#print the confusion matrix
print(confusion_matrix(y_test,predictions))

#Print the classification report
print(classification_report(y_test,predictions))

#ROC curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

y_true=sorted(y_test)
y_score=sorted(predictions)
# Compute fpr, tpr, thresholds and roc auc
fpr, tpr, thresholds = roc_curve(y_true, y_score)
roc_auc = auc(y_true, y_score)

# Plot ROC curve
plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate or (1 - Specifity)')
plt.ylabel('True Positive Rate or (Sensitivity)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")

# Compare Algorithms
from pandas import read_csv
from matplotlib import pyplot
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
# split data into X and y
X = features_columns_df.iloc[:,0:15]
Y = features_columns_df.iloc[:,15]
# prepare models
models = []
models.append(( ' LR ' , LogisticRegression()))
models.append(( ' LDA ' , LinearDiscriminantAnalysis()))
models.append(( ' KNN ' , KNeighborsClassifier()))
models.append(( ' CART ' , DecisionTreeClassifier()))
models.append(( ' NB ' , GaussianNB()))
models.append(( ' SVM ' , SVC()))
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
    kfold = KFold(n_splits=10, random_state=7)
    cv_results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# boxplot algorithm comparison
fig = pyplot.figure()
fig.suptitle( ' Algorithm Comparison ' )
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

