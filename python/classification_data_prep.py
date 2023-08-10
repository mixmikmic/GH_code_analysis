from modules import *

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import sklearn
import missingno as msno

plt.style.use('seaborn-dark')
plt.rcParams['figure.figsize'] = (10, 6)

conn = dbConnect()
objects = dbTableToDataFrame(conn, 'cb_objects')
degrees = dbTableToDataFrame(conn, 'cb_degrees')
people = dbTableToDataFrame(conn, 'cb_people')
conn.close()

#look at missing values of objects
msno.matrix(objects)

#in degrees and offices 'object_id' is matched to 'id' in objects
#I am going to create an 'object_id' in objects
ids = objects['id'].values
objects['object_id'] = ids

#function to convert variable to fill data
def convert_null(column, df, positive, negative, pos_val = False):
    """
    description: create a two class dummby variable for a dataframe with None values
    
    inputs:
        column: String, must be name of column in df
        df: Pandas Data Frame
        postive: String/Boolean/Integer, what to call non-none value in dumby variable
        negative: String/Boolean/Integer, what to call non value in dumby variable
        pos_val: Boolean (optional), if True then positive keep their own value
        
    output:
        dumby vector
    """
    if type(column) != str:
        raise ValueError("column and newname must be a string")
    if type(df) != pd.core.frame.DataFrame:
        raise ValueError("df must be pandas data frame")
    col = df[column]
    bools = col.isnull().values
    if pos_val == True:
        dumby = []
        for i in range(len(bools)):
            if bools[i] == True:
                dumby.append(negative)
            else:
                dumby.append(col.values[i])
        return(dumby)
    dumby = [positive if not status else negative for status in bools]
    return(dumby)

#change variables of responses:
closed = convert_null('closed_at', objects, 'Yes', 'No', False)
objects['closed'] = closed

#make dumby variable for funding: Yes if they got funding, no if not
funding = convert_null('first_funding_at', objects, 'Yes', 'No')
objects['had_funding'] = funding

#make variable for investment: Yes if they got investment, no if not
investment = convert_null('invested_companies', objects, 'Yes', 0.0, pos_val = True)
objects['num_investment'] = investment

#get number of relationships
num_relationships = convert_null('relationships', objects, 'Yes', 0.0, pos_val = True)
objects['num_relationships'] = num_relationships

#get number of milestones
num_milestones = convert_null('milestones', objects, 'Yes', 0.0, pos_val = True)
objects['num_milestones'] = num_milestones

#deal with logo sizes
width = convert_null('logo_width', objects, 'Yes', 0.0, pos_val = True)
objects['logo_width'] = width

height = convert_null('logo_height', objects, 'Yes', 0.0, pos_val = True)
objects['logo_height'] = height

def keep_variables(df, variables, object_id = True):
    """
    description: create a two class dummby variable for a dataframe with None values
    
    inputs:
        df: Pandas Data Frame from where columns are selected
        variables: a list of strings of column names of variables to select
        object_id: (optional) boolean variable, if False do not keep object_id, else add object_id to variables
        
    output:
        Pandas Data Frame of variables we want to keep
    """
    #check inputs are good
    if type(variables) != list:
        raise TypeError("variables must be a list of strings")
    if not all(isinstance(item, str) for item in variables):
        raise ValueError("all items in variables must be strings")
    if type(df) != pd.core.frame.DataFrame:
        raise ValueError("df must be pandas data frame")
    if object_id:
        variables = ['object_id'] + variables
    return df[variables]

#keep variables from people
people['name'] = people['affiliation_name']
vars_people = keep_variables(people, ['birthplace', 'first_name', 'last_name', 'name'])

#keep variables from degrees
vars_degrees = keep_variables(degrees, ['degree_type', 'institution', 'subject'])

#keep variables from objects
vars_objects = keep_variables(objects, ['closed', 'status', 'name', 'category_code', 'had_funding', 'num_investment', 'num_relationships', 'num_milestones', 'logo_height', 'logo_width', 'region'])

def drop_duplicates(dfs_info):
    """
    description: create a two class dummby variable for a dataframe with None values
    
    inputs:
        dfs_info: List tuples. First item is data frame, second is subset info for drop_dublic, last is keep infor for drop_duplicate 
        
    output:
        list of pandas dfs equal to length of list
    """
    dfs = []
    for item in dfs_info:
        cur_df = item[0]
        cur_subset = item[1]
        cur_keep = item[2]
        edit_df = cur_df.drop_duplicates(subset = cur_subset, keep = cur_keep)
        dfs.append(edit_df)
    return(dfs)

#set up function
first = (vars_people, 'name', 'first')
second = (vars_degrees, 'object_id', 'last')
dfs_info = [first, second]

#edit dfs
vars_people, vars_degrees = drop_duplicates(dfs_info)

def multi_merge(dfs, times, on = 'object_id', leave_out = []):
    """
    description: merge multiple data frame
    
    inputs:
        dfs: List of Pandas Data Frame to be merged
        times: number of seperate merges
        on: (optional) String, column to be merged on
        leave_out: (optional) list of 2 item tuples. Each tuple's first item is a data frame from the df list and the second item is a list of columns to leave out of merge.
        
    output:
        Pandas Data Frame of meged dfs
    """
    if type(dfs) != list:
        raise TypeError("dfs must be a list of Pandas dfs")
    if not all(isinstance(item, pd.core.frame.DataFrame) for item in dfs):
        raise ValueError("all items in dfs must be Pandas DFs")
        
    if len(leave_out) != 0:
        if type(leave_out) is not list or not all(isinstance(t, tuple) for t in leave_out):
            raise TypeError("leave_out must be a list of tuples")
        for item in leave_out:
            df_to_alter = item[0]
            cols = item[1]
            if df_to_alter not in dfs:
                raise ValueError("data frams in leave_out must be in multi_merge")
            df_alterted = df_to_alter.drop(cols, 1)
            index = dfs.index(df_to_alter)
            dfs[index] = df_alterted
            
    first = dfs[0]
    rest = dfs[1:]
    for i in range(len(rest)):
        cur_df = rest[i]
        first = first.merge(cur_df, on = on)
    return(first)

#merge the data frames
deg_pep = vars_degrees.merge(vars_people, on = 'object_id')

#drop object id
#df1 = obj_off.drop('object_id', 1)
df1 = vars_objects.drop('object_id', 1)
df2 = deg_pep.drop('object_id', 1)

dat = df1.merge(df2, on = 'name')
dat.shape

#replace missing value
rep_na = dat.replace("", np.nan)
rep_na = rep_na.replace("unknown", np.nan)

#look at emptiness of data
msno.matrix(rep_na)

data = rep_na.dropna(axis = 0, how = 'any')
data.shape

data.head()

def make_plots(response, feat, data, kind):
    
    """
    description: helper function to view view reponse and feature relationship
    inputs:
        reponse: string name of response variable in data
        feat: string name of feature in data
        data: dataframe
        kind: kind of plot
    output:
        baxplot that is saved with title
    
    """
    
    end = ".".join([kind, "png"])
    fig = "_".join([response, end])
    title = " ".join([feat, response, kind])
    fig_end = "_".join([feat, fig])
    save_fig = "/".join(['results', fig_end])
    
    if (kind == 'box'):
        plt.figure()
        sns.boxplot(data = data, y = feat, x = response)

        
    if (kind == 'bar'):
        fig = "_".join([response, 'bar.png'])
        group = data.groupby([feat, response])[[response, feat]].size().unstack()
        
        resp_freqs = group.sum(axis = 0).values
        cat_code_resp_rel = group / resp_freqs
        
        cat_freqs = cat_code_resp_rel.sum(axis = 1).values
        cat_code_rel = cat_code_resp_rel.div(cat_freqs, axis  = 0)

        cat_code_rel.plot(kind = 'bar')

        
    plt.title(title)
    plt.savefig(save_fig)
        

features = ['num_relationships', 'num_milestones', 'num_investment', 'logo_height', 'logo_width']
for f in features:
    make_plots(response = 'status', feat = f, data = data, kind = 'box')
    make_plots(response = 'closed', feat = f, data = data, kind = 'box')

data.groupby(['category_code', 'status'])[['status', 'category_code']].size().unstack().plot(kind = 'bar')
plt.show()

cat_code = data.groupby(['category_code', 'status'])[['status', 'category_code']].size().unstack()

resp_freqs = cat_code.sum(axis = 0).values
cat_code_resp_rel = cat_code / resp_freqs

cat_code_resp_rel.plot(kind = 'bar')
plt.show()

features = ['category_code', 'had_funding', 'region', 'degree_type', 'institution', 'subject']
for f in features:
    make_plots(response = 'status', feat = f, data = data, kind = 'bar')
    make_plots(response = 'closed', feat = f, data = data, kind = 'bar')

data.head()

#save data
data.to_hdf('results/classification_data.h5', 'classification_data')

