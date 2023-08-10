import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
get_ipython().magic('matplotlib notebook')

# load data and name the column names
column_name =  ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
       's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
       's15', 's16', 's17', 's18', 's19', 's20', 's21' ]
train_FD001 = pd.read_table("./CMAPSSData/train_FD001.txt",header=None,delim_whitespace=True)
train_FD002 = pd.read_table("./CMAPSSData/train_FD002.txt",header=None,delim_whitespace=True)
train_FD003 = pd.read_table("./CMAPSSData/train_FD003.txt",header=None,delim_whitespace=True)
train_FD004 = pd.read_table("./CMAPSSData/train_FD004.txt",header=None,delim_whitespace=True)
train_FD001.columns = column_name
train_FD002.columns = column_name
train_FD003.columns = column_name
train_FD004.columns = column_name

for data in ['train_FD00' + str(i) for  i in range(1,5)]:
    # have a look at the info of each data file
    eval(data).info()

def compute_rul_of_one_id(train_FD00X_of_one_id):
    '''
    输入train_FD001的一个engine_id的数据，输出这些数据对应的RUL（剩余寿命），type为list
    '''
    max_cycle = max(train_FD00X_of_one_id['cycle'])  # 故障时的cycle
    rul_of_one_id = max_cycle - train_FD00X_of_one_id['cycle']
    return rul_of_one_id.tolist()

def compute_rul_of_one_file(train_FD00X):
    '''
    输入train_FD001，输出一个list'''
    rul = []
    # 循环train中，''engine_id''这一列的每一种id值
    for id in set(train_FD00X['engine_id']):
        rul.extend(compute_rul_of_one_id(train_FD00X[train_FD00X['engine_id'] == id]))
    return rul

# 为4个data增加RUL列
for data_file in ['train_FD00' + str(i) for  i in range(1,5)]:
    # have a look at the info of each data file
    eval(data_file)['RUL'] = compute_rul_of_one_file(eval(data_file))

# 重新设置index， 使四个data的index能衔接上
train_FD001.index = range(20631)
train_FD002.index = range(20631,20631+53759)
train_FD003.index = range(20631+53759,20631+53759+24720)
train_FD004.index = range(20631+53759+24720,20631+53759+24720+61249)

# 将四个data拼接到一起，并设置hierarchical index : ['FD001', 'FD002', 'FD003', 'FD004']
frames = [train_FD001, train_FD002, train_FD003, train_FD004]
train = pd.concat(frames, keys = ['FD001', 'FD002', 'FD003', 'FD004'])

train.loc['FD001'][train.loc['FD001']['engine_id'] == 1]

train.to_csv('train_FD001_to_4')

train_all = pd.read_csv('train_FD001_to_4', index_col =[0,1])

train_all

