import pandas as pd

repo = pd.read_csv('Repositories.csv')
repo.head()               

repo2 = repo.drop('Month, Day, Year of Date',1)

repo2 = repo2.groupby(['Worker ID','Repo']).sum()
repo2 =repo2.unstack()
repo2.columns = repo2.columns.droplevel()
repo2 = repo2.fillna(0)
repo2.head()

repo2.to_csv('repo.csv')

repo.dtypes

repo['Month, Day, Year of Date'] = repo['Month, Day, Year of Date'].astype('str')

from datetime import datetime
date = repo['Month, Day, Year of Date'].tolist()
dataup = []
for i in range(len(date)):
    x = datetime.strptime(date[i], '%m/%d/%Y')
    dataup.append(x)

month = []
for d in range(len(dataup)):
    m = dataup[d].strftime('%m')
    month.append(m)
repo['month'] = month
repo.head()

repo = repo.drop('Month, Day, Year of Date',1)

repo_big = repo.loc[repo['Repo'].isin([139])]

repo_big.to_csv('repo_big.csv')

repo_07 = repo.loc[repo['month'] == '07']
repo_08 = repo.loc[repo['month'] == '08']
repo_09 = repo.loc[repo['month'] == '09']

repo_08 = repo_08.drop('month',1)
repo_07 = repo_07.drop('month',1)
repo_09 = repo_09.drop('month',1)

repo.head()

repo_07 = repo_07.groupby(['Worker ID','Repo']).sum()
repo_07 =repo_07.unstack()
repo_07.columns = repo_07.columns.droplevel()
repo_07 = repo_07.fillna(0)
repo_07.head()

repo_07.to_csv('repo_07.csv')

repo_08 = repo_08.groupby(['Worker ID','Repo']).sum()
repo_08 =repo_08.unstack()
repo_08.columns = repo_08.columns.droplevel()
repo_08 = repo_08.fillna(0)
repo_08.to_csv('repo_08.csv')
repo_08.head()

repo_09 = repo_09.groupby(['Worker ID','Repo']).sum()
repo_09 =repo_09.unstack()
repo_09.columns = repo_09.columns.droplevel()
repo_09 = repo_09.fillna(0)
repo_09.to_csv('repo_09.csv')
repo_09.head()

Repobyworker = Repo_Effort.groupby(['Worker ID','Repo']).sum()
Repounstack = Repobyworker.unstack()
Repounstack.columns = Repounstack.columns.droplevel()
Repounstack = Repounstack.fillna(0)
Repounstack.head()

Repounstack = Repobyworker.unstack()
Repounstack.columns = Repounstack.columns.droplevel()

Repounstack = Repounstack.fillna(0)
Repounstack.head()

Repounstack.to_csv('Repo_matrix.csv')

Task_Effort = Task_Effort.drop('Task id',1)

Task_Effort.head()

Task_Effort = Task_Effort.dropna()

Task_Effort.head()

Date = Task_Effort.Date.tolist()

month = []
for i in range(len(Date)):
    x = Date[i].split('/')[1]
    month.append(x)
for i in range(len(Date)):
    month[i] = month[i].strip('0')

Task_Effort['month'] = month

Task_Effort.head()

Task_Effort['Raw Effort']

Task_Effort = Task_Effort.convert_objects(convert_numeric=True)

Task_sum = Task_Effort.groupby(['Worker ID','Repo','month']).sum()

Task_sum.reset_index(level=0, inplace=True)

Task_sum.reset_index(level=0, inplace=True)

Task_sum.reset_index(level=0, inplace=True)

Task_sum.head()
Task_sum.columns = ['month','Repo','Worker_ID','Raw Effort']

Task_sum.head()

Task_sum.to_csv('edges_Task.csv')

nodes_Task = pd.DataFrame()
nodes_Task['id'] = Task_sum.Repo.unique().tolist() + Task_sum.Worker_ID.unique().tolist()
nodes_Task['label'] = Task_sum.Repo.unique().tolist() + Task_sum.Worker_ID.unique().tolist()
nodes_Task['Cat'] = ['Repo'] * len(Task_sum.Repo.unique()) + ['Worker'] * len(Task_sum.Worker_ID.unique())
nodes_Task.head()

nodes_Task.to_csv('nodes_Task.csv')

task.columns = ['Bug_id','Date','Worker_id','Task_id']
task = task.dropna()

task.isnull().any()

task.head()

task.to_csv('for_tablea.csv')

taskclean  = taskclean.sort_values('Worker_id')

len(task)/len(task.groupby('Worker_id').count())

task_clean = task.groupby(['Worker_id','Task_id']).count().drop('Bug_id',1)

task_clean.head()

task_clean.reset_index(level=0, inplace=True)
task_clean.head()

task_clean.reset_index(level=0, inplace=True)
task_clean.head()

nodes = pd.DataFrame()
nodes['id'] = task_clean.Task_id.unique().tolist() + task_clean.Worker_id.unique().tolist()
nodes['label'] = task_clean.Task_id.unique().tolist() + task_clean.Worker_id.unique().tolist()
nodes['Cat'] = ['Task']*len(task_clean.Task_id.unique()) + ['Worker'] * len(task_clean.Worker_id.unique())
nodes.head()

nodes.to_csv('nodes.csv')

task_clean = task_clean.unstack()
task_clean.columns = task_clean.columns.droplevel()

task_clean = task_clean.fillna(0)

task_clean.head()

task_clean.to_csv('R_matrix.csv')

task_clean = pd.read_csv('R_matrix.csv') 

task_clean.head()

task_clean.to_csv('edge.csv')

taskpivot = taskclean.sort_values(['Worker_id','Date'])

taskpivot = taskclean.reset_index().drop('index',1)

taskpivot.head()

taskprepared = taskpivot.groupby(['Worker_id','Task_id']).count()

taskprepared.columns = ['Bug_cnt','Date_cnt']
taskprepared = taskprepared.drop('Bug_cnt',1)

taskprepared.head()



taskprepared.Task_id.unique()

taskprepared.Worker_id.unique()





nodes = pd.DataFrame()
nodes['id'] = list(taskprepared.Task_id.unique()) + list(taskprepared.Worker_id.unique())
nodes['label'] = list(taskprepared.Task_id.unique()) + list(taskprepared.Worker_id.unique())
nodes['Attribute'] = ['Task'] * len(taskprepared.Task_id.unique()) + ['Worker'] * len(list(taskprepared.Worker_id.unique()))
nodes.to_csv('nodes.csv')

taskprepared.reset_index(level=0, inplace=True)
taskprepared.reset_index(level=0, inplace=True)

#taskprepared = taskprepared.drop('index',1)
#taskprepared = taskprepared.drop('Date',1)
taskprepared.columns = ['Task_id','Worker_id','weight/days']
taskprepared.to_csv('Task_edges.csv')

nodes = pd.DataFrame()
nodes['id'] = list(taskprepared.Task_id.unique())+ list(taskprepared.Worker_id.unique())
nodes['type'] = ['Task'] * len(taskprepared.Task_id.unique()) + ['Worker'] * len(taskprepared.Worker_id.unique())

nodes.to_csv('Task_nodes.csv')

taskprepared = taskprepared.unstack()
taskprepared.columns = taskprepared.columns.droplevel()

taskprepared.head()

taskprepared.columns

taskprepared.head()

taskprepared.fillna(0).sort_index().to_csv('task_matrix.csv')

taskprepared.fillna(0).head()

taskclean.head()

Worker_id = []
Task_id = []
worker_ori = taskclean.Worker_id.tolist()

for i in range(len(worker_ori)):
    x = 'W'+str(worker_ori[i])
    Worker_id.append(x)
taskclean['Worker_id'] = Worker_id

task_ori = taskclean.Task_id.tolist()
for i in range(len(task_ori)):
    x = 'T'+str(task_ori[i])
    Task_id.append(x)
taskclean['Task_id'] = Task_id

taskclean.head()

taskclean.Worker_id.unique()

worker_list = ['Worker'] * len(taskclean.Worker_id.unique())
worker = pd.DataFrame()
worker['id'] = taskclean.Worker_id.unique()
worker['type'] = worker_list
worker.head()

task_list = ['Task'] * len(taskclean.Task_id.unique())
TASK = pd.DataFrame()
TASK['id'] = taskclean.Task_id.unique()
TASK['type'] = task_list
TASK.head()

node = pd.concat([worker,TASK],ignore_index = True)

node.to_csv('Task_matrix_node.csv')

