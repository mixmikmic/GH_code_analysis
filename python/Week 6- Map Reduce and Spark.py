# This opens the full text of Moby Dick file into a file object


import codecs
with codecs.open('./datasets/moby_full','r',encoding='utf8') as f:
    text = f.read()

# Now, we create a TextBlob object from the file 

from textblob import TextBlob

full_text = TextBlob(text)

# See the text in the object

full_text

# Count noun phrases. Note it will take some time depending on your computer.

get_ipython().magic('time serial = full_text.np_counts')

print 'Length of noun phrases is {}'.format(len(serial))
print 'Sum of noun phrase counts is {}'.format(sum(serial.values()))

# Start IPython Parallel in notebook and check for workers

from ipyparallel import Client
c = Client()

# Now, let's check the client to make sure all four workers have started

print 'These are the currently active worker ids:{}'.format(c.ids)

# Assign all workers to a view

dview=c[:]

text_list = ['moby25a', 'moby25b', 'moby25c', 'moby25d']

@dview.parallel(block=True)
def read_texts_parallel(text):
    from textblob import TextBlob
    import codecs
    with codecs.open('./datasets/{}'.format(text[0]),'r',encoding='utf8') as f:
        text = f.read()
    blob = TextBlob(text)
    counts = blob.np_counts
    return dict(counts)    

from collections import Counter

def map_reduce(texts):
    # This effectively maps the iterable list of texts to the function on each worker
    mapped_text = read_texts_parallel(texts)
    # This takes the returned map results and combines them in the notebook process
    reduced = reduce(lambda x, y:Counter(x) + Counter(y), mapped_text)
    return reduced


get_ipython().magic('time map_reduced = map_reduce(text_list)')

print 'Length of noun phrases is {}'.format(len(map_reduced))
print 'Sum of noun phrase counts in {}'.format(sum(map_reduced.values()))

set(serial).difference(set(map_reduced))

