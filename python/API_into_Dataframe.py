from json import loads
import pandas as pd

## Load sample json data from NPR's API
json_obj = loads(open("npr_api_sample.json").read())

## Get to know the data
for story in json_obj['list']['story']:

    print 'ID: ' + story['id']
    
    print 'TITLE: ' + story['title']['$text']

    print 'DATE: ' + story['storyDate']['$text'] + '\n'

## For our visualization, we're interested in the story category tags, which are nested in the json

for story in json_obj['list']['story']:
    for parent in story['parent']:
        print story['id'], parent['title']['$text']

## Build the data frame by creating a list of dictionaries, then converting the list of dictionaries into a data frame

##First data frame: one row per distinct story category

## Create an empty list
dicts_list = []

## Fill the list with dictionaries -- each dictionary will be a row in our dataframe

for story in json_obj['list']['story']:
    for parent in story['parent']:
        d = {
            'id': story['id'],
            'title': story['title']['$text'],
            'category': parent['title']['$text']    
        }
        dicts_list.append(d)

## Convert the list of dictionaries into a pandas dataframe

df = pd.DataFrame(dicts_list, columns=('id', 'title', 'category'))

df.head(5)

##Second data frame: One row per story, with one column containing a list of all the story's categories

## Create an empty list
dicts_reshape = []

## Fill the list with dictionaries -- each dictionary will be a row in our dataframe

for story in json_obj['list']['story']:
    categories_list = []
    d = {
        'id': story['id'],
        'title': story['title']['$text']
        }
    for parent in story['parent']:
        category = parent['title']['$text']
        categories_list.append(category)   
        d['category'] = categories_list
        d['top_category'] = []
    dicts_reshape.append(d)

## Convert the list of dictionaries into a pandas dataframe

df_reshape = pd.DataFrame(dicts_reshape, columns=('id', 'title', 'category'))

df_reshape.head()

df_reshape.to_json('npr_dataframe_sample.json')



