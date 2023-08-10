import re
import pandas as pd

with open('data/raw/8k-data/GOOG.txt', 'r') as f:
    data = f.read()

# Split into documents
documents = data.replace('<DOCUMENT>', '').split('</DOCUMENT>')

len(documents)

reg_FILE = re.compile('FILE:(.*)')
reg_TIME = re.compile('TIME:(.*)')
reg_EVENTS = re.compile('EVENTS:(.*)')
reg_ITEMS = re.compile('ITEM:(.*)')

def factory_parser(compiled_reg):
    return lambda d: map(lambda x: x.replace(',', '').strip(), re.findall(compiled_reg, d))

get_file = factory_parser(reg_FILE)
get_time = factory_parser(reg_TIME)
get_events = factory_parser(reg_EVENTS)
get_items = factory_parser(reg_ITEMS)

META_LINES = ['FILE', 'TIME', 'EVENTS', 'ITEM']

def get_text(doc):
    lines = [ ln for ln in doc.split('\n') if ln != ' ' ]
    g_lines = [ ln for ln in lines if not any(map(lambda x: ln.startswith(x), META_LINES)) ]
    return '\n'.join( g_lines )

# Parse documents
parsed_data = {
    'file': map(get_file, documents),
    'time': map(get_time, documents),
    'events': map(get_events, documents),
    'items': map(get_items, documents),
    'text': map(get_text, documents)
 }

struct_data = pd.DataFrame.from_dict(parsed_data)

struct_data.head()

struct_data.to_csv('data/parsed/8k-data/GOOG.csv')



