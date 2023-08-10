# Pandas is a python library that is used for data manipulation and analysis -- good for numbers + time series.
    # Pandas gives us some extra data structures (arrays are data structures, for example) which is nice
    # We are calling Pandas pd by using the "as" -- locally, we know Pandas as pd
    # Helpful Links:
        # https://en.wikipedia.org/wiki/Pandas_(software)
        # http://pandas.pydata.org/ 
import pandas as pd

# Sharepa is a python client for  browsing and analyzing SHARE data specifically using elasticsearch querying.
    # We can use this to aggregate, graph, and analyze the data. 
    # Helpful Links:
        # https://github.com/CenterForOpenScience/sharepa
        # https://pypi.python.org/pypi/sharepa
from sharepa import ShareSearch

#When we say from X import Y, we are saying "of all the things in this python library, import only this
from sharepa.helpers import pretty_print

description_search = ShareSearch()

# exists -- a type of query, will accept a lucene query string
    # Lucene supports fielded data. When performing a search you can either specify a field, or use the default field. 
    # The field names and default field is implementation specific.
# field = description -- This lucene query string will find all documents that don't have a description
description_search = description_search.query(
    'exists', 
    field='description',
)

# here we are aggregating all the entries by source
description_search.aggs.bucket(
    'sources',  # Every aggregation needs a name
    'significant_terms',  # There are many kinds of aggregations
    field='sources',  # We store the source of a document in its type, so this will aggregate by source
    min_doc_count=0,
    percentage={}, # Will make the score value the percentage of all results (doc_count/bg_count)
    size=0
)

description_results = description_search.execute()

# Creates a dataframe using Pandas (what we call pd) that aggregates the results
description_dataframe = pd.DataFrame(description_results.aggregations.sources.to_dict()['buckets'])

# We will add our own "percent" column to make things clearer
description_dataframe['percent'] = (description_dataframe['score'] * 100)

# Let's set the source name as the index, and then drop the old column
description_dataframe = description_dataframe.set_index(description_dataframe['key'])
description_dataframe = description_dataframe.drop('key', 1)

# Finally, we'll show the results!
description_dataframe

# Note: Uncomment the following lines if running locally:

description_dataframe.to_csv('SHARE_Counts_with_Descriptions.csv')
description_dataframe.to_excel('SHARE_Counts_with_Descriptions.xlsx')

# this is a simple list
names = ["Susan Jones", "Ravi Patel"]

#this is a 
name_search = ShareSearch()

# We are searching the entire SHARE dataset for each item in the list we called name, i.e. Susan Jones and Ravi Patel
for name in names:
    name_search = name_search.query(
        {
            "bool": {
                "should": [
                    {
                        "match": {
                            "contributors.full_name": {
                                "query": name, 
                                "operator": "and",
                                "type" : "phrase"
                            }
                        }
                    }
                ]
            }
        }
    )

# We are putting all the results into a new list called name_results
# name_search is our original list, and .execute() is a built-in function (one that the library provides, and we
    # don't have to write) that puts the results of the loop above into a new list
name_results = name_search.execute()

# Prints out the number of documents that have those 
print('There are {} documents with contributors who have any of those names.'.format(name_search.count()))

# Just visual queues for us to make it more readable
print('Here are the first 10:')
print('---------')

# Loops over the list called "name_results" and prints out 10
for result in name_results:
    print(
        '{} -- with contributors {}'.format(
            result.title,
            [contributor.full_name for contributor in result.contributors]
        )
    )

name_search.aggs.bucket(
    'sources',  # Every aggregation needs a name
    'terms',  # There are many kinds of aggregations, terms is a pretty useful one though
    field='sources',  # We store the source of a document in its type, so this will aggregate by source
    size=0,  # These are just to make sure we get numbers for all the sources, to make it easier to combine graphs
    min_doc_count=1
)

# We are putting all the results into a new list called name_results
# name_search is our original list, and .execute() is a built-in function (one that the library provides, and we
    # don't have to write) that puts the results of the loop above into a new list
name_results = name_search.execute()

# We are aggregating these into a DataFrame from Pandas (which we called pd)
pd.DataFrame(name_results.aggregations.sources.to_dict()['buckets'])



