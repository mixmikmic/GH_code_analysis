get_ipython().magic('matplotlib inline')

from pprint import pprint
import matplotlib.pyplot as plt

from tethne.readers import wos
datapath = '/Users/erickpeirson/Downloads/datasets/wos'
corpus = wos.read(datapath)

print 'The primary index field for the Papers in my Corpus is "%s"' % corpus.index_by

corpus.indexed_papers.items()[0:10]    # We'll just show the first ten Papers, for the sake of space.

corpus.indexed_papers['WOS:000321911200011']

otherCorpus = wos.read(datapath, index_by='doi')

print 'The primary index field for the Papers in this other Corpus is "%s"' % otherCorpus.index_by

i = 0
for doi, paper in otherCorpus.indexed_papers.items()[0:10]:
    print '(%i) DOI: %s \t ---> \t Paper: %s' % (i, doi.ljust(30), paper)
    i += 1

print 'The following Paper fields have been indexed: \n\n\t%s' % '\n\t'.join(corpus.indices.keys())

for citation, papers in corpus.indices['citations'].items()[7:10]:   # Show the first three, for space's sake.
    print 'The following Papers cite %s: \n\n\t%s \n' % (citation, '\n\t'.join(papers))

papers = corpus.indices['citations']['CARLSON SM 2004 EVOL ECOL RES']  # Who cited Carlson 2004?
print papers
for paper in papers:
    print corpus.indexed_papers[paper]

corpus.index('authorKeywords')

for keyword, papers in corpus.indices['authorKeywords'].items()[6:10]:   # Show the first three, for space's sake.
    print 'The following Papers contain the keyword %s: \n\n\t%s \n' % (keyword, '\n\t'.join(papers))

corpus.index('date')

for date, papers in corpus.indices['date'].items()[-11:-1]:    # Last ten years.
    print 'There are %i Papers from %i' % (len(papers), date)

corpus.distribution()[-11:-1]    # Last ten years.

plt.figure(figsize=(10, 3))
start = min(corpus.indices['date'].keys())
end = max(corpus.indices['date'].keys())
X = range(start, end + 1)
plt.plot(X, corpus.distribution(), lw=2)
plt.ylabel('Number of Papers')
plt.xlim(start, end)
plt.show()

corpus['WOS:000309391500014']

corpus[('authorKeywords', 'LIFE')]

corpus[['WOS:000309391500014', 'WOS:000306532900015']]

corpus[('authorKeywords', ['LIFE', 'ENZYME GENOTYPE', 'POLAR AUXIN'])]

papers = corpus[('date', range(2002, 2013))] # range() excludes the "last" value.
print 'There are %i Papers published between %i and %i' % (len(papers), 2002, 2012)

corpus.features.items()

featureset = corpus.features['authors']
for k, author in featureset.index.items()[0:10]:
    print '%i  -->  "%s"' % (k, ', '.join(author)) # Author names are stored as (LAST, FIRST M).

featureset = corpus.features['authors']
for author, k in featureset.lookup.items()[0:10]:
    print '%s  -->  %i' % (', '.join(author).ljust(25), k)

featureset = corpus.features['authors']
for k, count in featureset.documentCounts.items()[0:10]:
    print 'Feature %i (which identifies author "%s") is found in %i documents' % (k, ', '.join(featureset.index[k]), count)

featureset.features.items()[0]

corpus.index_feature('authorKeywords')
corpus.features.keys()

featureset = corpus.features['authorKeywords']
for k, count in featureset.documentCounts.items()[0:10]:
    print 'Keyword %s is found in %i documents' % (featureset.index[k], count)

featureset.features['WOS:000324532900018']    # Feature for a specific Paper.

plt.figure(figsize=(10, 3))
years, values = corpus.feature_distribution('authorKeywords', 'DIVERSITY')
start = min(years)
end = max(years)
X = range(start, end + 1)
plt.plot(years, values, lw=2)
plt.ylabel('Papers with DIVERSITY in authorKeywords')
plt.xlim(start, end)
plt.show()

