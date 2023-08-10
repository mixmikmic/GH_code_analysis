import numpy as np
import scrapePM as spm
from bs4 import BeautifulSoup
import urllib

# scraping the cognitive atlas webpage for terms
CAurl = urllib.urlopen('http://www.cognitiveatlas.org/concepts/').read()
soup = BeautifulSoup(CAurl, 'html.parser')
concepts = soup.find_all('span', attrs={'class': 'concept'})
terms = []
for c in concepts:
    terms.append(str(c.text.strip()).lower())
    
terms.append('') # append empty term to calculate total article count

base_phrase = 'AND("cognitive"OR"cognition")'
term_counts_cog = spm.scrape_terms(terms, base_phrase=base_phrase, fieldkey='TIAB')

base_phrase = ('AND('+
                '("fmri"OR"neuroimaging")OR'+
                '("pet"OR"positron emission tomography")OR'+
                '("eeg"OR"electroencephalography")OR'+
                '("meg"OR"magnetoencephalography")OR'+
                '("ecog"OR"electrocorticography")OR'+
                '("lfp"OR"local field potential")OR'+
                '("erp"OR"event related potential")OR'+
                '("single unit"OR"single-unit"OR"single neuron")OR'+
                '("calcium imaging")'
                ')')
term_counts_neumet = spm.scrape_terms(terms, base_phrase=base_phrase, fieldkey='TIAB')

base_phrase = 'AND("neural"OR"neuroscience")'
term_counts_neu = spm.scrape_terms(terms, base_phrase=base_phrase, fieldkey='TIAB')

base_phrase = None
term_counts_gen = spm.scrape_terms(terms, base_phrase=base_phrase, fieldkey='TIAB')

# save out sorted term count list for PubMed
import csv
# save term frequency to file
with open('./data/term_counts_pubmed.csv', 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['term', 'all', 'cog', 'neu', 'neu_methods'])
    for t in sorted(term_counts_gen.keys()):
        writer.writerow([t, term_counts_gen[t], term_counts_cog[t],term_counts_neu[t],term_counts_neumet[t]])

