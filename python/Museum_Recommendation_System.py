import pandas as pd
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
from fancyimpute import KNN

museum_df = pd.read_csv("dummified_df.csv")
museum_df.columns.tolist()

idx_to_drop = [0,1,2,4,5,6,7,9,14,15] # drop non-numeric columns besides MuseumName
museum_df = museum_df.drop(museum_df.columns[idx_to_drop], axis=1)
museum_df.columns.tolist()

# impute missing value using knn imputation
# k is determined by the one yielding decent prediction for classification 'Rating' in 'TripAdvisor_Rating_Prediction.ipynb'
no_name_df = museum_df.drop(museum_df.columns[[1]], axis = 1)
no_name_df.columns.tolist()

# impute missing value using knn imputation
# k is determined by the one yielding decent prediction for classification 'Rating' in 'TripAdvisor_Rating_Prediction.ipynb'
no_name_df = museum_df.drop(museum_df.columns[[1]], axis = 1)
X_filled_knn = KNN(k = 40).complete(no_name_df)
length = no_name_df.shape[0]
imputed_df = pd.DataFrame(data = X_filled_knn,
                          index= range(0,length),
                          columns = no_name_df.columns)

imputed_df.head()

# merge museum name with imputed dataframe
merged_df = pd.concat([museum_df['MuseumName'], imputed_df], axis=1)
# imputed_df.to_csv('imputed_df_no_name.csv')
merged_df.to_csv('./app/data/imputed_df_with_name.csv')
merged_df.describe()

def get_museum_lst(target_museum_input):
    '''get the museum lst from input'''
    return target_museum_input.split(';')[1:]

def get_master_srt_lst(museum_lst):
    '''concatenate all top five lists for museums in museum_lst'''
    master_srt_lst = []
    for m in museum_lst:
         master_srt_lst += get_top_five_for_one(m)
    return master_srt_lst

def sort_list(lst):
    '''sort the nested list based on the second item in list'''
    sorted_lst = sorted(lst, key=lambda x: x[1], reverse = True) 
    return sorted_lst

def get_top_five_for_one(target_museum):
    '''get top five museum and consine similarity for one musuem'''
    target_idx = museum_df[museum_df['MuseumName'] == target_museum].index.tolist()[0]
    input_vec = np.array(imputed_df.iloc[target_idx]).reshape(1, -1)
    nrow = imputed_df.shape[0]
    cos_sim = []
    for i in range(nrow):
        # reshapre the row into a vector
        vec = np.array(imputed_df.iloc[i]).reshape(1, -1)
        # compute and store consine similarity along with musuem name
        cos_sim.append([museum_df['MuseumName'][i], cosine_similarity(input_vec, vec)[0][0]])
    top_five  = sort_list(cos_sim)
    return top_five[1:6] # ignore the top one since it's the target musuem itself

def lst_to_dic(lst):
    '''convert lst into dictionary'''
    dic = {}
    for i in lst:
        dic[i[0]] = i[1]
    return dic

def to_json(name, dic):
    '''write dictionary to json file'''
    filename = name + '.json'
    with open(filename, 'w') as f:
        json.dump(dic, f)

def get_sorted_dic(lst):
    dic = {}
    for idx, item in enumerate(lst):
        dic[idx+1] = [item[0], item[1]]
    return dic
        
def exclude_selected(museum_lst, srt_lst):
    return [x for x in srt_lst if x[0] not in museum_lst]

museum_df = pd.read_csv("./app/data/imputed_df_with_name.csv")
museum_df = museum_df.drop(museum_df.columns[0], axis=1)
imputed_df = museum_df.drop(museum_df.columns[[0,4,5]], axis=1)

museum_df.head()

def get_unique_recom(master_srt_lst):
    unique_name = list(set([i[0]for i in master_srt_lst]))
    uni_lst = []
    for i in master_srt_lst:
        if i[0] in unique_name:
            uni_lst.append([ i[0],i[1] ])
            unique_name.pop(unique_name.index(i[0]))

    return uni_lst

target_museum_input = ';British Museum;The Metropolitan Museum of Art'
# target_museum_input = ';Science Museum'
museum_lst = get_museum_lst(target_museum_input)
master_srt_lst = get_master_srt_lst(museum_lst)
uni_lst = get_unique_recom(master_srt_lst)
sorted_lst = sort_list(uni_lst)
top_lst = exclude_selected(museum_lst, sorted_lst)
sorted_dic = get_sorted_dic(top_lst)
to_json('./app/data/testing_top_five', sorted_dic)
# sorted_dic

sorted_dic

top_lst  

# features that are included in calculating the cosine similarity
print 'number of features:', len(imputed_df.columns)
imputed_df.columns

pd.DataFrame({'colname':imputed_df.columns.values})



