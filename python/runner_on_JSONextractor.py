# Add the path to the slack-pack/code/ folder in order to be able to import nlp module
import sys, os

NLP_PATH = '/'.join(os.path.abspath('.').split('/')[:-1]) + '/'
sys.path.append(NLP_PATH)

from nlp.text import extractor as xt
from nlp.text.message import Message

PATH_TO_DATA = 'nlp/data/general/2016-05-06.json'

dataxt = xt.JSONExtractor(PATH_TO_DATA)

messages = dataxt.get_messages()

for msg in messages:
    print msg.text
    print ''



