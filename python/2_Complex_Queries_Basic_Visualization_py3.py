# Json library parses JSON from strings or files. The library parses JSON into a Python dictionary or list. 
    # It can also convert Python dictionaries or lists into JSON strings. 
    # https://docs.python.org/2.7/library/json.html
import json

# Requests library allows you to send organic, grass-fed HTTP/1.1 requests, no need to manually add query strings 
    # to your URLs, or to form-encode your POST data. Docs: http://docs.python-requests.org/en/master/
import requests

# This takes the URL and puts it into a variable (so we only need to ever reference this variable, 
    # and so we don't have to repeat adding this URL when we want to work with the data)
SHARE_API = 'https://staging-share.osf.io/api/search/abstractcreativework/_search'

# A helper function that will use the requests library, pass along the correct headers, and make the query we want
def query_share(url, query):
    headers = {'Content-Type': 'application/json'}
    data = json.dumps(query)
    return requests.post(url, headers=headers, data=data, verify=False).json()

# A function that prints out the results in a numbered list
def print_numbered_results(results):
    print(
        'There are {} total results and {} results on this page'.format(
            results['hits']['total'],
            len(results['hits']['hits'])
        )
    )
    print('---------------')    
    for result in enumerate(results['hits']['hits']):
        print('{}. {}'.format(result[0] + 1, result[1]['_source']['title']))

# We are setting up a query for items in the SHARE dataset that have the keyword "frogs"
basic_query = {
    "query": {
        "query_string": {
            "query": "frogs"
        }
    }
}

#this puts the results of querying SHARE_API with what we outlined in basic_query (frogs)
query_results = query_share(SHARE_API, basic_query)

#print out the numbered list of the results
print_numbered_results(query_results)

# We are setting up a query for items in the SHARE dataset that have the keyword "frogs" but limiting the 
    # results to 20 items
basic_query = {
    "query": {
        "query_string": {
            "query": "frogs"
        }
    },
    "size": 20
}

# this puts the results of querying SHARE_API with what we outlined in basic_query (frogs)
query_results = query_share(SHARE_API, basic_query)

# print out the numbered list of the results
print_numbered_results(query_results)

# We are setting up a query for items in the SHARE dataset that have the keyword "frogs" 
basic_query = {
    "query": {
        "query_string": {
            "query": "frogs"
        }
    }
}

# creates a list of the first 10 results from the search query "frogs"
query_results = query_share(SHARE_API, basic_query)

# print the results of the search we made in a numbered list
print_numbered_results(query_results)

# make it visually pretty and readable for us humans
print('------------------------------------------')
print('*** Making another query for the next page ***')
print('*** These next titles will be different! ***')
print('------------------------------------------')

basic_query['from'] = 10  # Add the 'from' parameter to the query to pick up at the next page of results

# creates a list of the next 10 results 
query_results = query_share(SHARE_API, basic_query)

# print the results of the search we made in a numbered list
print_numbered_results(query_results)

# this is a function that helps us print the lists with pagination between results (10, next 10, etc.)
def print_numbered_sharepa_results(search_obj):
    results = search_obj.execute()
    print(
        'There are {} total results and {} results on this page'.format(
            search_obj.count(),
            len(results.hits)
        )
    )
    print('---------------')    
    for result in enumerate(results.hits):
        print('{}. {}'.format(result[0] + 1, result[1]['title']))

# Sharepa is a python client for  browsing and analyzing SHARE data specifically using elasticsearch querying.
    # We can use this to aggregate, graph, and analyze the data. 
    # Helpful Links:
        # https://github.com/CenterForOpenScience/sharepa
        # https://pypi.python.org/pypi/sharepa
    # here, we import the specific function from Sharepa called ShareSearch and pretty_print
from sharepa import ShareSearch
from sharepa.helpers import pretty_print

# we are creating a new local search!
frogs_search = ShareSearch()

# this sets up what we will actually search for -- keyword "frogs"
frogs_search = frogs_search.query(
    'query_string',
    query='frogs'
)

# print the results of the search we made for "frogs" keyword in a numbered list
print_numbered_sharepa_results(frogs_search)

# print the 10th - 20th results of the search we made for "frogs" keyword in a numbered list
print_numbered_sharepa_results(frogs_search[10:20])

# this aggregates the number of documents that have no tags, per source, using query boolean (not tags) while also 
    # grabbing all sources in the aggregations: sources below
missing_tags_aggregation = {
    "query": {
        "bool": {
            "must_not": {
                "exists": {
                    "field": "tags"
                  }
            }
        }
    },
    "aggregations": {
        "sources": {
            "terms": {
                "field": "sources", # A field where the SHARE source is stored                
                "min_doc_count": 0, 
                "size": 0  # Will return all sources, regardless if there are results
            }
        }
    }
}

# puts all the items without tags into a list
results_without_tags = query_share(SHARE_API, missing_tags_aggregation)

# counts the number of items without tags
missing_tags_counts = results_without_tags['aggregations']['sources']['buckets']

# this prints out the number of documents with missing tags, separated by sources
for source in missing_tags_counts:
    print('{} has {} documents without tags'.format(source['key'], source['doc_count'], ))

# this does the same as above, but adds the percentage of documents missing tags from the total number of documents
    # per source
no_tags_query = {
    "query": {
        "bool": {
            "must_not": {
                "exists": {
                    "field": "tags"
                  }
            }
        }
    },
    "aggs": {
        "sources":{
            "significant_terms":{
                "field": "sources", # A field where the SHARE source is stored                
                "min_doc_count": 0, 
                "size": 0,  # Will return all sources, regardless if there are results
                "percentage": {} # This will make the "score" parameter a percentage
            }
        }
    }
}

# creates a list of the documents with no tags
docs_with_no_tags_results = query_share(SHARE_API, no_tags_query)

#creates a dictionary that shows the results of the search along with the aggregations we outlined above
docs_with_no_tags = docs_with_no_tags_results['aggregations']['sources']['buckets']

# this prints out the number of documents with missing tags, separated by sources, with the percentage that makes up
    # the total number of documents for each source
for source in docs_with_no_tags:
    print(
        '{}% (or {}/{}) of documents from {} have no tags'.format(
            format(source['score']*100, '.2f'),
            source['doc_count'],
            source['bg_count'],
            source['key']
        )
    )

# yay! creating another new search! 
no_language_search = ShareSearch()

# this sets up our search query: all documents with no language field
no_language_search = no_language_search.query(
    'bool',
    must_not={"exists": {"field": "language"}}
)

no_language_search.aggs.bucket(
    'sources',  # Every aggregation needs a name
    'significant_terms',  # There are many kinds of aggregations
    field='sources',  # We store the source of a document in its type, so this will aggregate by source
    min_doc_count=1,
    percentage={},
    size=0
)

# print (prettily!) the items grabbed the search that have no language field
pretty_print(no_language_search.to_dict())

# here we grab the results of items that have no language field + their sources and significant terms
aggregated_results = no_language_search.execute()

# this prints out the percentage of items that don't have a language field
for source in aggregated_results.aggregations['sources']['buckets']:
    print(
        '{}% of documents from {} do not have language'.format(
            format(source['score']*100, '.2f'),
            source['key'] 
        )
    )

# creating another new search!
top_tag_search = ShareSearch()

# this sets up our search query with the aggregations
top_tag_search.aggs.bucket(
    'tagsTermFilter',  # Every aggregation needs a name
    'terms',  # There are many kinds of aggregations
    field='tags',  # We store the source of a document in its type, so this will aggregate by source
    min_doc_count=1,
    exclude= "of|and|or",
    size=10
)

# pretty_print(top_tag_search.to_dict())

# this executes the search as we've outlined it above
top_tag_results_executed = top_tag_search.execute()

# this places the results of the search into this dictionary
top_tag_results = top_tag_results_executed.aggregations.tagsTermFilter.to_dict()['buckets']

# this prints out our search results (prettily)
pretty_print(top_tag_results)

# Pandas is a python library that is used for data manipulation and analysis -- good for numbers + time series.
    # Pandas gives us some extra data structures (arrays are data structures, for example) which is nice
    # We are calling Pandas pd by using the "as" -- locally, we know Pandas as pd
    # Helpful Links:
        # https://en.wikipedia.org/wiki/Pandas_(software)
        # http://pandas.pydata.org/ 
import pandas as pd

# this transforms our results from the cell above into a dataframe
top_tags_dataframe = pd.DataFrame(top_tag_results)

#this prints out our dataframe -- looks like a nice table!
top_tags_dataframe

# Matplot lib is a a python 2D plotting library which produces publication quality figures in a variety of hardcopy
    # formats and interactive environments across platforms. 
    # Read more about Matplotlib here: http://matplotlib.org/
from matplotlib import pyplot

# this is used specifically with iPython notebooks to display the matplotlib chart or graph in the notebook
get_ipython().magic('matplotlib inline')

#this takes our results dictionary from the cell above and plots them into a bar chart
top_tags_dataframe.plot(kind='bar', x='key', y='doc_count')

# this prints out the bar chart we just made!
pyplot.show()

# from the SHAREPA library, we import this function to transform a bucket into a dataframe so we can plot it!
from sharepa import bucket_to_dataframe

# creating a new search! 
all_results = ShareSearch()

# creating our search query!
all_results = all_results.query(
    'query_string', # Type of query, will accept a lucene query string
    query='*', # This lucene query string will find all documents that don't have tags
    analyze_wildcard=True  # This will make elasticsearch pay attention to the asterisk (which matches anything)
)

# Lucene supports fielded data. When performing a search you can either specify a field, or use the default field. 
all_results.aggs.bucket(
    'sources',  # Every aggregation needs a name
    'terms',  # There are many kinds of aggregations, terms is a pretty useful one though
    field='sources',  # We store the source of a document in its type, so this will aggregate by source
    size=0,  # These are just to make sure we get numbers for all the sources, to make it easier to combine graphs
    min_doc_count=0
)

# this executes our search!
all_results = all_results.execute()

# this uses that function we imported above to transform our aggregated search into a dataframe so we can plot it!
all_results_frame = bucket_to_dataframe('# documents by source', all_results.aggregations.sources.buckets)

# this sorts the dataframe by the number of documents by source (descending order)
all_results_frame_sorted = all_results_frame.sort(ascending=False,  columns='# documents by source')

# this creates a bar chart that displays the first 30 results 
all_results_frame_sorted[:30].plot(kind='bar')

# Creating a pie graph using the first 10 items in the data frame with no legend
all_results_frame_sorted[:10].plot(kind='pie', y="# documents by source", legend=False)



