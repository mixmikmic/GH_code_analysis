import networkx
import PageRank
import json
import operator
import sys
import glob
import os

sys.path.append("..")
from all_functions import *

path_to_json = '../PageRank_OUTPUT'
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
Page_rank_dict={}
for js in json_files:
    with open(os.path.join(path_to_json, js)) as json_file:
        new_json=json.load(json_file)
    Page_rank_dict.update(new_json)

word_count_places = read_data('../Analyzed_data//final_titles_ids_all_status.csv')
places_population = read_data('../Analyzed_data//final_titles_ids_all_clean_population.csv')

places_population_1=places_population[['wof:id', '_population']]

word_count_places_pop = word_count_places.join(places_population_1.set_index(['wof:id']), on='wof:id', how = 'left')

word_count_places_clean=word_count_places_pop[word_count_places_pop['spell_check']=='OK']
key_set=set(Page_rank_dict.keys())

wordcount_pagerank=[]
for index, row in word_count_places_clean.iterrows():
    name = row['wk:page']
    if name in key_set:
        score = Page_rank_dict[name]
        row['Page_rank_score'] = score
    else:
        pass
    wordcount_pagerank.append(row)
    
df_wordcount_pagerank = pd.DataFrame(wordcount_pagerank)

df_wordcount_pagerank_1=df_wordcount_pagerank[['wof:id','placetype','Page_rank_score','_population','wordcount' ]]

word_count_places_sorted = word_count_places_pop.sort('wordcount', ascending=False)
sorted_list = word_count_places_sorted[(word_count_places_sorted['spell_check']=='OK')&(word_count_places_sorted['placetype']=='locality')]

cities_unique = find_unique(sorted_list,'name')
data = range(len(cities_unique))
cities_unique['ran'] =data

NE_ranking = read_data('C:\Users\Olga\Documents\MAPZEN_data\Projects\Wiki\\NE_ranking.csv')

cities_unique_rank = []
for index, row in cities_unique.iterrows():
    if row['ran']<=9000:
        row['ranking_10'] = 1
    elif 9000<row['ran']<=18000:
        row['ranking_10'] = 2
    elif 18000<row['ran']<=27000:
        row['ranking_10'] = 3
    elif 27000<row['ran']<=36000:
        row['ranking_10'] = 4
    elif 36000<row['ran']<=45000:
        row['ranking_10'] = 5
    elif 45000<row['ran']<=54000:
        row['ranking_10'] = 6
    elif 54000<row['ran']<=63000:
        row['ranking_10'] = 7
    elif 63000<row['ran']<=72000:
        row['ranking_10'] = 8
    elif 72000<row['ran']<=81000:
        row['ranking_10'] = 9
    elif row['ran']>81000:
        row['ranking_10'] = 10
    cities_unique_rank.append(row)
    

cities_unique_rank_df=pd.DataFrame(cities_unique_rank)

cities_unique_rank_df_join=cities_unique_rank_df.join(NE_ranking.set_index(['NAME']), on='name', how='left')

cities_unique_rank_df_join_grouped = cities_unique_rank_df_join.groupby('ranking_10')

cities_unique_rank_df_join_grouped.get_group(3)

df_wordcount_pagerank_placetype = df_wordcount_pagerank_1.groupby('placetype')

df_wordcount_pagerank_1['placetype'].unique()



macrocount=df_wordcount_pagerank_placetype.get_group('country')

macrocount['percent_word']=macrocount['wordcount']/max(macrocount['wordcount'])

macrocount

with open('..\PageRank_OUTPUT\Page_Rank_macrocounty.json', 'r') as outfile:
       dictionary_countries_links =  json.load(outfile)



