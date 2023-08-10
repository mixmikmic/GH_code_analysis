from glove import list_corpora, GloVe

# Let's check which we have available to load:
list_corpora()

get_ipython().run_cell_magic('time', '', "glove_100 = GloVe('glove.6B.100d.txt')")

glove_100('Horse')  # don't worry about caps, it takes care of that underneath :)

# we can pass a whole text as well, and it will output the centroid of all the words
glove_100('My name is Will')



