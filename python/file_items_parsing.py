import parsing_tools as prs

prs.parse_files(folder='../data/8K_gz/', output_folder='../data/parsed/')



import glob
import pandas as pd
import numpy as np

def get_itemset(table):
    return set(reduce(lambda x, y: x + y, table['items'].apply( eval ).values))

class ItemMap(object):
    
    NAME_TEMPL = 'item_{i}'
    
    def __init__(self):
        self.ITEM_MAP = {}
    
    def __getitem__(self, item):
        if item not in self.ITEM_MAP:
            self.ITEM_MAP[item] = self.NAME_TEMPL.format(len(self.ITEM_MAP))
        
        return self.ITEM_MAP[item]

csv_files = glob.glob('../data/parsed/*.csv')

itemset = set()

for _filepath in csv_files:
    with open(_filepath, 'rb') as f:
        _data = pd.read_csv(f)

    itemset.update( get_itemset(_data) )

itemset



import glob

# Get parsed files
csv_files = glob.glob('../data/parsed/*.csv')
raw_files = glob.glob('../data/8K-gz/*.gz')



print 'START!'

import gzip
import glob
import re
import pandas as pd

from collections import Counter
from nltk.corpus import stopwords

# Stopwords
STOPWORDS = set(stopwords.words('english'))

## Variable-parsing functions

META_LINES = ['FILE', 'TIME', 'EVENTS', 'ITEM']

# regex templates
reg_FILE = re.compile('FILE:(.*)')
reg_TIME = re.compile('TIME:(.*)')
reg_EVENTS = re.compile('EVENTS:(.*)')
reg_ITEMS = re.compile('ITEM:(.*)')

# A parsing_function generator
def factory_parser(compiled_regex):
    return lambda d: map(lambda x: x.replace(',', '').strip(), re.findall(compiled_regex, d))

get_file = factory_parser(reg_FILE)
get_time = factory_parser(reg_TIME)
get_events = factory_parser(reg_EVENTS)
get_items = factory_parser(reg_ITEMS)

def get_text(doc):
    lines = [ ln for ln in doc.split('\n') if ln != ' ' ]
    g_lines = [ ln for ln in lines if not any(map(lambda x: ln.startswith(x), META_LINES)) ]
    return '\n'.join( g_lines )

# Bag of Words parser
def process_bow(text):
    """ Returns a Counter with the Bag of Words of the text inputted """
    _text = text.lower().replace('TEXT:', '').replace('Table of Contents', '')

    # Split into words and remove common (NLTK) stopwords
    words = filter(lambda x: x not in STOPWORDS, _text.split())

    return Counter(words)


## Full folder parser function

def parse_files(folder, output_folder='../data/parsed/'):
    """ Unzips all .gz files found in the specified folder """

    files = glob.glob( '{f}/*.gz'.format(f=folder) )

    print( '\n\033[34mReading {} zipped files ...\033[0m'.format(len(files)) )

    for gzip_file in files:

        with gzip.open(gzip_file, 'rb') as f_in:
            _data = f_in.read()

        
        # Split into documents
        documents = _data.replace('<DOCUMENT>', '').split('</DOCUMENT>')

        # Parse documents and create dataframe
        parsed_data = {
            'file': list(map(get_file, documents)),
            'time': list(map(get_time, documents)),
            'events': list(map(get_events, documents)),
            'items': list(map(get_items, documents)),
            'text': list(map(get_text, documents))
        }

        data = pd.DataFrame.from_dict(parsed_data).iloc[:-1, :]

    
        # Generate final (parsed) features
        data['bow'] = data.text.apply( lambda x: str(process_bow(x)) )
        data['date'] = data.time.apply( lambda x: pd.datetime(year=int(x[0][:4]),
                                                              month=int(x[0][4:6]),
                                                              day=int(x[0][6:8])) )
        data['orig_file'] = data.file.apply( lambda x: x[0] )

        # replace the '.gz' with '.csv'
        output_file = gzip_file.replace('.gz', '.csv').split('/')[-1]
        data[['date', 'bow', 'items', 'text', 'orig_file' ]].to_csv(output_folder+output_file, index=False)

        break

    print( '\033[34mDone! Located in:  \033[32m{}\033[0m'.format(output_folder) )

get_ipython().magic('pdb on')

parse_files('../data/8K_gz')











