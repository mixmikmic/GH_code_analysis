import networkx
import PageRank
import json
import operator
import os
get_ipython().magic('matplotlib inline')

import mpld3
mpld3.enable_notebook()

import sys
sys.path.append("..")
from all_functions import *

path_to_json = '..\OUTPUTS_linkshere'
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
all_links_dict={}
for js in json_files:
    with open(os.path.join(path_to_json, js)) as json_file:
        new_json=json.load(json_file)
    all_links_dict.update(new_json)

wof_ids = read_data("..\Analyzed_data\\final_titles_ids_all_status.csv")

mapping_dic_wof_name={}
for index, row in wof_ids.iterrows():
    key = row['wk:page']
    value = str(int(row['wof:id']))
    mapping_dic_wof_name.update({key:value})

mapping_dic_name_wof={}
for index, row in wof_ids.iterrows():
    key = str(int(row['wof:id']))
    value = row['wk:page']
    mapping_dic_name_wof.update({key:value})

dictionary_wof_links={}
for key, value in all_links_dict.iteritems():
    try:
        a = key.encode('utf8')
        new_key=mapping_dic_wof_name[a]
        dictionary_wof_links.update({new_key:value})
    except KeyError:
        pass

import pickle
with open('..\Jupyter_notebooks_with_analysis\\names_do_not_want', 'rb') as f:
    names_do_not_want = pickle.load(f)

wof_do_not_want = []
for item in names_do_not_want:
    wof = int(item)
    wof_do_not_want.append(wof)
    
flattened = []
for x in wof_do_not_want:
    if type(x) is list:
        flattened.extend(x)
    else:
        flattened.append(x)
        
names_set=set(flattened)

dictionary_wof_links_clean={}
for key, value in dictionary_wof_links.iteritems():
    if key in names_set:
        pass
    else:
        dictionary_wof_links_clean.update({key:value})

with open('..\Analyzed_data\\linkshere_wof_clean.json', 'w') as outfile:
    json.dump(dictionary_wof_links_clean, outfile)

dictionary_names_links_clean={}
for key, value in dictionary_wof_links_clean.iteritems():
    try:
        new_key=mapping_dic_name_wof[key]
        dictionary_names_links_clean.update({new_key:value})
    except KeyError:
        pass

with open('..\Analyzed_data\\linkshere_names_clean.json', 'w') as outfile:
    json.dump(dictionary_names_links_clean, outfile)



set_path_to_WOF_metafiles = 'C:\Users\Olga\Documents\MAPZEN_data\whosonfirst-data\meta'

wof_continents_latest=read_data(set_path_to_WOF_metafiles+"\wof-continent-latest.csv")
wof_country_latest=read_data(set_path_to_WOF_metafiles+"\wof-country-latest.csv")
wof_borough_latest=read_data(set_path_to_WOF_metafiles+"\wof-borough-latest.csv")
wof_campus_latest=read_data(set_path_to_WOF_metafiles+"\wof-campus-latest.csv")
wof_county_latest=read_data(set_path_to_WOF_metafiles+"\wof-county-latest.csv")
wof_locality_latest=read_data(set_path_to_WOF_metafiles+"\wof-locality-latest.csv")
wof_macrocounty_latest=read_data(set_path_to_WOF_metafiles+"\wof-macrocounty-latest.csv")
wof_macrohood_latest=read_data(set_path_to_WOF_metafiles+"\wof-macrohood-latest.csv")
wof_macroregion_latest=read_data(set_path_to_WOF_metafiles+"\wof-macroregion-latest.csv")
wof_marinearea_latest=read_data(set_path_to_WOF_metafiles+"\wof-marinearea-latest.csv")
wof_microhood_latest=read_data(set_path_to_WOF_metafiles+"\wof-microhood-latest.csv")
wof_neighbourhood_latest=read_data(set_path_to_WOF_metafiles+"\wof-neighbourhood-latest.csv")
wof_ocean_latest=read_data(set_path_to_WOF_metafiles+"\wof-ocean-latest.csv")
wof_planet_latest=read_data(set_path_to_WOF_metafiles+"\wof-planet-latest.csv")
wof_region_latest=read_data(set_path_to_WOF_metafiles+"\wof-region-latest.csv")
wof_empire_latest=read_data(set_path_to_WOF_metafiles+"\wof-empire-latest.csv")

frames=[wof_continents_latest,wof_country_latest,wof_borough_latest,wof_campus_latest,wof_county_latest,wof_locality_latest,
       wof_macrocounty_latest, wof_macrohood_latest,wof_macroregion_latest,wof_marinearea_latest,wof_microhood_latest,wof_neighbourhood_latest,
        wof_ocean_latest,wof_planet_latest,wof_region_latest,wof_empire_latest]
all_wof = pd.concat(frames)

wiki_original_names=read_data("..\Analyzed_data\\final_titles_ids_all_status.csv")

wiki_original_names_clean=wiki_original_names[wiki_original_names['spell_check']=='OK']

wiki_original_names_clean_places=wiki_original_names_clean[wiki_original_names_clean['placetype'].notnull()]

wiki_original_names_clean_places=wiki_original_names_clean_places[['wof:id','wk:page']]

all_wof_names = all_wof.join(wiki_original_names_clean_places.set_index(['wof:id']), on='id', how = 'left' )

all_wof_names_notnull=all_wof_names[all_wof_names['wk:page'].notnull()]

all_wof_names_grouped = all_wof_names_notnull.groupby('placetype')

wof_country_names = set(all_wof_names_grouped.get_group('country')['wk:page'])
wof_borough_names = set(all_wof_names_grouped.get_group('borough')['wk:page'])
wof_campus_names = set(all_wof_names_grouped.get_group('campus')['wk:page'])
wof_county_names = set(all_wof_names_grouped.get_group('county')['wk:page'])
wof_locality_names = set(all_wof_names_grouped.get_group('locality')['wk:page'])
wof_macrocounty_names = set(all_wof_names_grouped.get_group('macrocounty')['wk:page'])
wof_macrohood_names = set(all_wof_names_grouped.get_group('macrohood')['wk:page'])
wof_macroregion_names = set(all_wof_names_grouped.get_group('macroregion')['wk:page'])
wof_marinearea_names = set(all_wof_names_grouped.get_group('marinearea')['wk:page'])
wof_microhood_names = set(all_wof_names_grouped.get_group('microhood')['wk:page'])
wof_neighbourhood_names = set(all_wof_names_grouped.get_group('neighbourhood')['wk:page'])
wof_region_names = set(all_wof_names_grouped.get_group('region')['wk:page'])

dictionary_countries_links={}
dictionary_borough_links={}
dictionary_campus_links={}
dictionary_county_links={}
dictionary_locality_links={}
dictionary_macrocounty_links={}
dictionary_macrohood_links={}
dictionary_macroregion_links={}
dictionary_marinearea_links={}
dictionary_microhood_links={}
dictionary_neighbourhood_links={}
dictionary_region_links={}

for key, value in dictionary_names_links_clean.iteritems():
    if key in wof_country_names:
        dictionary_countries_links.update({key:value})
    elif key in wof_borough_names:
        dictionary_borough_links.update({key:value})
    elif key in wof_campus_names:
        dictionary_campus_links.update({key:value})
    elif key in wof_county_names:
        dictionary_county_links.update({key:value})
    elif key in wof_locality_names:
        dictionary_locality_links.update({key:value})
    elif key in wof_macrocounty_names:
        dictionary_macrocounty_links.update({key:value})
    elif key in wof_macrohood_names:
        dictionary_macrohood_links.update({key:value})
    elif key in wof_macroregion_names:
        dictionary_macroregion_links.update({key:value})
    elif key in wof_marinearea_names:
        dictionary_marinearea_links.update({key:value})
    elif key in wof_microhood_names:
        dictionary_microhood_links.update({key:value})
    elif key in wof_neighbourhood_names:
        dictionary_neighbourhood_links.update({key:value})
    elif key in wof_region_names:
        dictionary_region_links.update({key:value})
    else:
        pass

with open('..\PageRank_OUTPUT\Page_Rank_countries_dict.json', 'w') as outfile:
    json.dump(dictionary_countries_links, outfile)

with open('..\PageRank_OUTPUT\\Page_Rank_campus_dict.json', 'w') as outfile:
    json.dump(dictionary_campus_links, outfile)
with open('..\PageRank_OUTPUT\\Page_Rank_county_dict.json', 'w') as outfile:
    json.dump(dictionary_county_links, outfile)
with open('..\PageRank_OUTPUT\\Page_Rank_macrocounty_dict.json', 'w') as outfile:
    json.dump(dictionary_macrocounty_links, outfile)
with open('..\PageRank_OUTPUT\\Page_Rank_macrohood_dict.json', 'w') as outfile:
    json.dump(dictionary_macrohood_links, outfile)
with open('..\PageRank_OUTPUT\\Page_Rank_marinearea_dict.json', 'w') as outfile:
    json.dump(dictionary_marinearea_links, outfile)
with open('..\PageRank_OUTPUT\\Page_Rank_microhood_dict.json', 'w') as outfile:
    json.dump(dictionary_microhood_links, outfile)
with open('..\PageRank_OUTPUT\\Page_Rank_region_dict.json', 'w') as outfile:
    json.dump(dictionary_region_links, outfile)

import itertools
def get_range(dictionary, begin, end):
      return dict(itertools.islice(dictionary.iteritems(), begin, end+1)) 

n=1
for i in range(0,len(dictionary_locality_links.keys()),1000):
    dictionary_locality_links_1=get_range(dictionary_locality_links,i,i+1000)
    with open('..\Page_Rank_data\\Page_Rank_locality_dict_%s.json' %n, 'w') as outfile:
        json.dump(dictionary_locality_links_1, outfile)
    n+=1

n=1
for i in range(0,len(dictionary_neighbourhood_links.keys()),1000):
    dictionary_neigh_links_1=get_range(dictionary_neighbourhood_links,i,i+1000)
    with open('..\Page_Rank_data\\Page_Rank_neighbourhood_dict_%s.json' %n, 'w') as outfile:
        json.dump(dictionary_neigh_links_1, outfile)
    n+=1



