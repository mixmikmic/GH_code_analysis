import pandas as pd
import numpy as np
from datetime import datetime
import requests
import re
import io
import urllib
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from collections import defaultdict
import pickle

# text cleaning imports
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

# filepath to twitter archive data.
twitter_json = r'C:\Users\aregel\Documents\springboard\Twitter-Politics\data_wrangle\data\twitter_01_01_16_to_11-16-17.json' 

# read the json data into a pandas dataframe
tweet_data = pd.read_json(twitter_json)
# set column 'created_at' to the index
tweet_data.set_index('created_at', drop=True, inplace= True)
# convert timestamp index to a datetime index
pd.to_datetime(tweet_data.index)

# function to identify hash tags
def hash_tag(text):
    return re.findall(r'(#[^\s]+)', text) 
# function to identify @mentions
def at_tag(text):
    return re.findall(r'(@[A-Za-z_]+)[^s]', text)

# tokenize all the tweet's text
tweet_data['text_tokenized'] = tweet_data['text'].apply(lambda x: word_tokenize(x))
# apply hash tag function to text column
tweet_data['hash_tags'] = tweet_data['text'].apply(lambda x: hash_tag(x))
# apply at_tag function to text column
tweet_data['@_tags'] = tweet_data['text'].apply(lambda x: at_tag(x))

# pickle tweet data so that loading and cleaning only needs to be done once
tweet_pickle_path = r'C:\Users\aregel\Documents\springboard\Twitter-Politics\data_wrangle\data\twitter_01_01_16_to_11-16-17.pickle'
tweet_data.to_pickle(tweet_pickle_path)

# Define the 2016, 2017 url that contains all of the Executive Office of the President's published documents
executive_office_url_2016 = r'https://www.federalregister.gov/index/2016/executive-office-of-the-president'
executive_office_url_2017 = r'https://www.federalregister.gov/index/2017/executive-office-of-the-president' 

# scrape all urls for pdf documents published 2016-01-01 to 2017-10-30 by the U.S.A. Executive Office
pdf_urls= []
for url in [executive_office_url_2016, executive_office_url_2017]:
    response = requests.get(url)
    pattern = re.compile(r'https:.*\.pdf')
    pdfs = re.findall(pattern, response.text)
    pdf_urls.append(pdfs)
    

# writes all of the pdfs to the data folder
start = 'data/'
end = '.pdf'
num = 0
for i in range(0,(len(pdf_urls))):
    for url in pdf_urls[i]:
        ver = str(num)
        pdf_path = start + ver + end
        r = requests.get(url)
        file = open(pdf_path, 'wb')
        file.write(r.content)
        file.close()
        num = num + 1

# function to convert pdf to text from stack overflow (https://stackoverflow.com/questions/26494211/extracting-text-from-a-pdf-file-using-pdfminer-in-python/44476759#44476759)
def convert_pdf_to_txt(path):
    rsrcmgr = PDFResourceManager()
    retstr = io.StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
    fp = open(path, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    password = ""
    maxpages = 0
    caching = True
    pagenos = set()

    for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages,
                                  password=password,
                                  caching=caching,
                                  check_extractable=True):
        interpreter.process_page(page)

    text = retstr.getvalue()

    fp.close()
    device.close()
    retstr.close()
    return text
# finds the first time the name of a day appears in the txt, and returns that name
def find_day(word_generator):
    day_list = ['Monday,', 'Tuesday,', 'Wednesday,', 'Thursday,', 'Friday,', 'Saturday,', 'Sunday,']
    day_name_dict = {'Mon':'Monday,', 'Tue':'Tuesday,','Wed':'Wednesday,','Thu':'Thursday,','Fri':'Friday,','Sat':'Saturday,','Sun':'Sunday,'}
    day_name = []
    for val in word_generator:
        if val in day_list:
            num_position = txt.index(val)
            day_name.append(txt[num_position] + txt[num_position + 1] + txt[num_position +2])
            break
            
    return day_name_dict[day_name[0]]
# takes text and returns the first date in the document
def extract_date(txt):
    word_generator = (word for word in txt.split())
    day_name = find_day(word_generator)
    txt_start = int(txt.index(day_name))
    txt_end = txt_start + 40
    date_txt = txt[txt_start:txt_end].replace('\n','')
    cleaned_txt = re.findall('.* \d{4}', date_txt)
    date_list = cleaned_txt[0].split()
    clean_date_list = map(lambda x:x.strip(","), date_list)
    clean_date_string = ", ".join(clean_date_list)
    date_obj = datetime.strptime(clean_date_string, '%A, %B, %d, %Y')
    return date_obj

start_path = r'C:/Users/aregel/Documents/springboard/Twitter-Politics/data_wrangle/data/'
end_path = '.pdf'
data_dict = defaultdict(list)
for i in range(0,559):
    file_path = start_path + str(i) + end_path
    txt = convert_pdf_to_txt(file_path)
    date_obj = extract_date(txt)
    data_dict[date_obj].append(txt)

tuple_lst = []
for k, v in data_dict.items():
    if v != None:
        for text in v:
            tuple_lst.append((k, text))
        

# create dataframe from list of tuples
fed_reg_dataframe = pd.DataFrame.from_records(tuple_lst, columns=['date','str_text'], index = 'date')

# tokenize all the pdf text
fed_reg_dataframe['token_text'] = fed_reg_dataframe['str_text'].apply(lambda x: word_tokenize(x))

# final dataframe
fed_reg_dataframe.head()

fed_reg_data = r'C:/Users/aregel/Documents/springboard/Twitter-Politics/data_wrangle/data/fed_reg_date_index.pickle'
fed_reg_dataframe.to_pickle(fed_reg_data)



