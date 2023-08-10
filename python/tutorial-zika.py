import numpy as np
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from pycproject.readctree import CProject
from pycproject.factnet import *
import os
from collections import Counter

get_ipython().magic('matplotlib inline')

def get_word_frequencies(cproject):
    words = Counter()
    for ctree in cproject.get_ctrees():
        if 'word' in ctree.results:
            for word in ctree.results['word']['frequencies']:
                words.update({word['word'], int(word['count'])})
    return words

def get_pub_years(cprojects):
    years_list = []
    final_years_list = []
    max_year = 0
    min_year = 3000
    
    for cproject in cprojects:
        years = Series()
        for ctree in cproject.get_ctrees():
            year = str(ctree.first_publication_date)[2:6]
            if int(year) < min_year:
                min_year = int(year)
            if int(year) > max_year:
                max_year = int(year)
            if year in years:
                years[year] += 1
            else:
                years[year] = 1
        years_list.append(years)

    for years in years_list:
        final_years = Series()
        
        for i in range(max_year - min_year + 1):
            if str(min_year + i) in years:
                final_years[str(min_year + i)] = years[str(min_year + i)]
            else: 
                final_years[str(min_year + i)] = 0
        
        final_years_list.append(final_years.sort_index())
    return final_years_list

def get_authors(cproject):
    authors = Counter()
    for ctree in cproject.get_ctrees():
        if 'authorList' in ctree.metadata:
            #print(ctree.metadata['authorList'][0]['author'])
            ctree_authors = ctree.metadata['authorList'][0]['author']
            for ctree_author in ctree_authors:
                if 'fullName' in ctree_author:
                    authors.update(ctree_author['fullName'])
    return authors

def get_journals(cproject):
    journals = Counter()
    for ctree in cproject.get_ctrees():
        if 'journalInfo' in ctree.metadata:
            #print(ctree.metadata['authorList'][0]['author'])
            ctree_journals = ctree.metadata['journalInfo'][0]['journal']
            for ctree_journal in ctree_journals:
                if 'title' in ctree_journal:
                    journals.update(ctree_journal['title'])
    return journals

def get_words(words, num_pubs):
    n_words = []
    for word, count in words.most_common(num_pubs):
        n_words.append((word, count))
    return n_words

zika = CProject("", "zika") # empty path value means we are already in the root directory
print(zika.size)

aedesaegypti = CProject("", "aedesaegypti") # empty path value means we are already in the root directory
print(aedesaegypti.size)

usutu = CProject("", "usutu") # empty path value means we are already in the root directory
print(usutu.size)

years = get_pub_years([zika, aedesaegypti, usutu]) # pass the CProjects you want to have a look at in a list

fig = plt.figure(figsize=(16, 12), dpi=300)

# draw subplot 1 = zika
ax1 = fig.add_subplot(3, 1, 1)
years[0].plot(kind='bar', ax=ax1)
ax1.set_title('Zika virus publications')

# draw subplot 2 = aedesaegytpi
ax2 = fig.add_subplot(3, 1, 2)
years[1].plot(kind='bar', ax=ax2)
ax2.set_title('Aedes aegypti publications')

# draw subplot 3 = usutu
ax3 = fig.add_subplot(3, 1, 3)
ax3.set_title('Usutu virus publications')
years[2].plot(kind='bar', ax=ax3)

# save plot
plt.savefig('timeline.svg')

zika_authors = get_authors(zika)

# num_authors = len(authors) # set to this, if you want to see all authors listed up. Be ware, this can lead to a very long list in the browser window.
num_authors = 20 # change the value to your needs

for author in zika_authors.most_common()[:num_authors]:
    print(author)

aedesaegypti_authors = get_authors(aedesaegypti)

# num_authors = len(authors) # set to this, if you want to see all authors listed up
num_authors = 20 # change the value to your needs

for author in aedesaegypti_authors.most_common()[:num_authors]:
    print(author)

usutu_authors = get_authors(usutu)

# num_authors = len(authors) # set to this, if you want to see all authors listed up
num_authors = 20 # change the value to your needs

for author in usutu_authors.most_common()[:num_authors]:
    print(author)

print('Number Zika authors', len(zika_authors))
print('Number Usutu authors', len(usutu_authors))
print('Matches: ', len(set(zika_authors).intersection(usutu_authors)))

zika_journals = get_journals(zika)

# num_authors = len(zika_journals) # set to this, if you want to see all authors listed up
num_journals = 20 # change the value to your needs

for journal in zika_journals.most_common()[:num_journals]:
    print(journal)

aedesaegypti_journals = get_journals(aedesaegypti)

# num_journals = len(aedesaegypti_journals) # set to this, if you want to see all authors listed up
num_journals = 20 # change the value to your needs

for journal in aedesaegypti_journals.most_common()[:num_journals]:
    print(journal)

usutu_journals = get_journals(usutu)

# num_journals = len(usutu_journals) # set to this, if you want to see all authors listed up
num_journals = 20 # change the value to your needs

for journal in usutu_journals.most_common()[:num_journals]:
    print(journal)

zika_words = get_word_frequencies(zika)

# num_words = len(zika_words) # set to this, if you want to see all authors listed up
num_words = 20 # change the number to your needs

for word in zika_words.most_common()[:num_words]:
    print(word)

aedesaegypti_words = get_word_frequencies(aedesaegypti)

# num_words = len(aedesaegypti_words) # set to this, if you want to see all authors listed up
num_words = 20 # change the number to your needs

for word in aedesaegypti_words.most_common()[:num_words]:
    print(word)

usutu_words = get_word_frequencies(usutu)

# num_words = len(usutu_words) # set to this, if you want to see all authors listed up
num_words = 20 # change the number to your needs

for word in usutu_words.most_common()[:num_words]:
    print(word)

# B_genus, genus_fact_graph, genus_paper_graph, genus_fact_nodes, genus_paper_nodes = create_network(aedesaegypti, "species", "genus")
# B_genus, genus_fact_graph, genus_paper_graph, genus_fact_nodes, genus_paper_nodes = create_network(usutu, "species", "genus")
B_genus, genus_fact_graph, genus_paper_graph, genus_fact_nodes, genus_paper_nodes = create_network(zika, "species", "genus")

start_with = 0 # enter a number here
how_many = 20 # # this will give us the next 10
degreeCent = nx.algorithms.degree_centrality(B_genus)
for node in sorted(degreeCent, key=degreeCent.get, reverse=True)[start_with:start_with+how_many]:
    print(len(B_genus.neighbors(node)), node)

how_many = 20 # # this will give us the next 10
degreeCent = nx.algorithms.degree_centrality(B_genus)
for node in sorted(degreeCent, key=degreeCent.get, reverse=True)[start_with:start_with+how_many]:
    print(len(B_genus.neighbors(node)), node)

# B_binomial, binomial_fact_graph, binomial_paper_graph, binomial_fact_nodes, binomial_paper_nodes = create_network(aedesaegypti, "species", "binomial")
# B_binomial, binomial_fact_graph, binomial_paper_graph, binomial_fact_nodes, binomial_paper_nodes = create_network(usutu, "species", "binomial")
B_binomial, binomial_fact_graph, binomial_paper_graph, binomial_fact_nodes, binomial_paper_nodes = create_network(zika, "species", "binomial")

start_with = 0 # enter a number here
how_many = 20 # # this will give us the next 10
degreeCent = nx.algorithms.degree_centrality(B_binomial)
for node in sorted(degreeCent, key=degreeCent.get, reverse=True)[start_with:start_with+how_many]:
    print(len(B_binomial.neighbors(node)), node)

# B_genussp, genussp_fact_graph, genussp_paper_graph, genussp_fact_nodes, genussp_paper_nodes = create_network(aedesaegypti, "species", "genussp")
# B_genussp, genussp_fact_graph, genussp_paper_graph, genussp_fact_nodes, genussp_paper_nodes = create_network(binomial, "species", "genussp")
B_genussp, genussp_fact_graph, genussp_paper_graph, genussp_fact_nodes, genussp_paper_nodes = create_network(zika, "species", "genussp")

start_with = 0 # enter a number here
how_many = 20 # # this will give us the next 10
degreeCent = nx.algorithms.degree_centrality(B_genussp)
for node in sorted(degreeCent, key=degreeCent.get, reverse=True)[start_with:start_with+how_many]:
    print(len(B_genussp.neighbors(node)), node)

start_with = 0 # pick a number between 1 and 50
how_many = 3 # choose the number of communities you want to plot between 1 and 5. More takes a lot of space in your notebook
subgraphs = sorted(nx.connected_component_subgraphs(genus_fact_graph), key=len, reverse=True)[start_with:start_with+how_many]
for idx, sg in enumerate(subgraphs):
    degreeCent = nx.algorithms.degree_centrality(sg)
    print(max(degreeCent, key=degreeCent.get))
    save_graph(sg, "orange", 'local-community-'+str(idx + 1)) # choose a color, e.g. red, blue, green, ...

my_species = "Wolbachia"
print(my_species in genus_fact_nodes)
print(my_species in binomial_fact_nodes)
print(my_species in genussp_fact_nodes)

print("Number of neighbors:", len(genus_fact_graph.neighbors(my_species)))
for idx, neighbor in enumerate(genus_fact_graph.neighbors(my_species)):
    print(idx+1, ':', neighbor)

print("Number of neighbors:", len(B_genus.neighbors(my_species)))
for idx, neighbor in enumerate(B_genus.neighbors(my_species)):
    print(idx+1, ':', neighbor)

get_ipython().run_cell_magic('capture', '', 'M, fact_graph, paper_graph, fact_nodes, paper_nodes = create_complete_graph(zika)')

plotMultipartiteGraph(M)







