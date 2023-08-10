import pandas as pd
import numpy as np
import jellyfish
import re
from nltk.corpus import stopwords

lobby_time = pd.read_csv('./data/lobbyistTimeRange.csv', parse_dates=[0,1])

ind_map = pd.read_csv('./data/industries_standardized.csv',header = None, index_col=0)
ind_map.columns = ['original','industry1','industry2','subcategory']
ind_map.industry1 = ind_map.industry1.apply(lambda x: representsInt(x))
ind_map.to_csv('./data/industries_standardized.csv')

t = lobby_time.Start[3]

t.month

merged = pd.merge(lobby_time, ind_map, how='left', left_on="Industry", right_on='original')
merged['month_of_hire'] = merged.Start.apply(lambda x: x.month)
merged['year_of_hire'] = merged.Start.apply(lambda x: x.year)

merged.head(3)

clients_by_time_industry.columns

clients_by_time_industry = merged.groupby(['industry1','year_of_hire','month_of_hire'],as_index=False).agg({'Client':len}).sort_values(['year_of_hire','month_of_hire'])

clients_by_time_industry.columns = ['industry', 'year_of_hire', 'month_of_hire', 'clients']

clients_by_time_industry.to_csv('./data/clients_by_time_industry.csv')

merged[merged.industry1.isnull()].shape

ind_map.columns

from difflib import SequenceMatcher

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def cross_product_apply(iteratible, function, symmetric = True):
    length = len(iteratible)
    results_matrix = np.zeros((length,length))
    
    for i in range(0,length):

        if symmetric == False:
            j_length = length
        else:
            j_length = i
            
        for j in range(0,j_length):
            results_matrix[i,j] = function(iteratible[i], iteratible[j])
        
    return results_matrix
    
    
    


def category_matcher(original_categories):
    # want to return n x 2 array of original industries and their matches
    #np.array([[1, 2, 3], [4, 5, 6]], np.int32)
    final_categories = [] #[""] * len(original_categories)
    sim_threshhold = .6
    

    for idx1, original_cat in enumerate(original_categories):

        if len(final_categories) == 0:
            '''
            if it's the first in the list, then a better match can't be found,
            and we keep the first title
            '''
            final_categories.append(original_cat)
        
        else:
            '''
            if it's not the first item, then we compare it to all existing items in the list,
            looking for a match.
            
            if one matches over the threshhold, then we choose that
            otherwise, we choose a new category
            '''
            
            max_sim = 0
            idx_of_max = -1
            
            for idx2, final_category in enumerate(final_categories):                
                sim_rank = similar(original_cat, final_category)

                if sim_rank > max_sim:
                    max_sim = sim_rank
                    idx_of_match = idx2
                        
            if max_sim < sim_threshhold:
                final_categories.append(original_cat)
                #final_categories[idx1] = original_cat
            else:
                final_categories.append(final_categories[idx_of_match])
                #final_categories[idx1] = final_categories[idx_of_match]

    return pd.DataFrame(original_categories,final_categories)

def representsInt(s):
    try: 
        int(s)
        return np.nan
    except ValueError:
        return s
    

        

def split_industries(industries): 
    bad_words=list(set(stopwords.words('english'))) + ['']
    splitted_industries = []
    for thing in industries:
        splitted = re.split(pattern='[/|,\s?|&|(?<!-)\s]', string=thing)
        splitted = [word for word in splitted if word not in bad_words]
        splitted_industries.append(splitted)
        
    return splitted_industries


def prop_in_common(iteratible1, iteratible2):
    denominator = len(iteratible1)
    common_count = 0
    if denominator !=0:
        numerator = len(np.intersect1d(iteratible1,iteratible2))
        return 1.0*numerator/denominator
    else:
        return 0.0
    

lobby_city = pd.read_csv('./data/lobbyistFromCity.csv', parse_dates=[0,1])

industries = lobby_city.Industry.fillna("").unique()

industries_split = split_industries(industries)

industry_str_sim_matrix = cross_product_apply(industries, jellyfish.jaro_winkler)

industry_list_sim_matrix = cross_product_apply(industries_split, prop_in_common, symmetric=False)

'''
given an n*n matrix, find the number of categories k
such that all n elements are in a category only 
with other elements that have a positive value in the corresponding field
and return those categories
'''

def categorizer(numpy_matrix, threshhold_value = 0):
    length = numpy_matrix.shape[0]
    cluster_candidates = []
    
    for i in range(0,length):
        
        cluster = []
        
        first_vector = numpy_matrix[i,:]
        match_index = first_vector.nonzero()[0]
        
        cluster += [i]
        
        for index in match_index:

            match_value = numpy_matrix[i,index]
            if match_value > threshhold_value:
                cluster += [index]
        
        print(cluster in cluster_candidates)
        cluster_candidates += set(cluster)
        #print(industries[cluster])
    
    return(len(set(cluster_candidates)))
    
    
    
#clusters = 
categorizer(industry_list_sim_matrix)

'''
Given these candidates, what will happen if we choose to minimize the number of single-element groups
'''
def choose_best_clusters(clusters):
    elements = set([x for x in clusters])

    


choose_best_clusters(clusters)

for i in range(0,30):
    vector = industry_str_sim_matrix[i,:]

    print(industries[i] + " : " + str(industries[(vector > .75).nonzero()]))

lob_cli = lobby_city.groupby(['Lobbyist','Client'])

lob_ind = lobby_city.groupby(['Lobbyist','Industry'])

len(array_of_industries)

lobby_data.head(20)



