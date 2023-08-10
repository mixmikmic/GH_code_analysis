from bs4 import BeautifulSoup as bsoup
import re
import os
import nltk
from nltk.collocations import *
from itertools import chain
import itertools
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import MWETokenizer

xml_file_path = "./xml_files"
patents_raw = {}





tokenizer = RegexpTokenizer(r'[a-zA-Z]{2,}') 
patents_tokenized = {}











all_words = list(chain.from_iterable(patents_tokenized.values()))































save_file = open("patent_student.txt", 'w')









save_file.close()



