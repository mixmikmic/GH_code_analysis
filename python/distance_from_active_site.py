import pandas 

# I already calculated these, so why not use this as a starting point?

df = pandas.read_csv( '/Users/alex/Documents/bagel-benchmark/data_sets/reference/distance_from_active_site.csv' )
ref = df[ [ 'name', 'distance' ] ] 
ref.set_index( 'name', inplace=True )  

ref.to_csv( 'reference.csv' ) 

get_ipython().system(' head reference.csv')

