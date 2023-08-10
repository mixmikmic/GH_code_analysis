from lxml import html as htmlparser
from os import makedirs
from os.path import basename, exists, join, splitext
from shutil import unpack_archive
import csv
import requests
RECORD_LAYOUT_URL = 'http://www.cde.ca.gov/ta/ac/ap/reclayoutApiAvg.asp'
RAW_DATA_ZIP_URL = 'http://www3.cde.ca.gov/researchfiles/api/14avgtx.zip'
RAW_DATA_DIR = join('data', 'schools', 'raw')
RAW_DATA_ZIP_FILENAME = join(RAW_DATA_DIR, basename(RAW_DATA_ZIP_URL))
# the text file has the same name as the zip, just different extension
RAW_DATA_TXT_FILENAME = splitext(RAW_DATA_ZIP_FILENAME)[0] + '.txt'
makedirs(RAW_DATA_DIR, exist_ok=True)

# save and extract the zip file to the raw directory
if not exists(RAW_DATA_ZIP_FILENAME):
    resp = requests.get(RAW_DATA_ZIP_URL)
    with open(RAW_DATA_ZIP_FILENAME, 'wb') as wf:
        wf.write(resp.content)

if not exists(RAW_DATA_TXT_FILENAME):
    unpack_archive(RAW_DATA_ZIP_FILENAME, extract_dir=RAW_DATA_DIR)

# prepare the extracted datafile and directory
EXTRACTED_DATA_DIR = join('data', 'schools', 'extracted')
# the extracted file has same basename but a .csv extension
EXTRACTED_DATA_FILENAME = join(EXTRACTED_DATA_DIR, '{}.csv'.format(splitext(basename(RAW_DATA_TXT_FILENAME))[0]))
makedirs(EXTRACTED_DATA_DIR, exist_ok=True)

# parse the Record Layout webpage
resp = requests.get(RECORD_LAYOUT_URL)
htmldoc = htmlparser.fromstring(resp.text)
# whatever
xa, xb = 0,0  # these represent the field boundaries
rows = htmldoc.xpath('//tr[td]')
field_defs = [tr.xpath('(td[2] | td[4]/div)/text()') for tr in rows]
for i, (fieldname, fieldlength) in enumerate(field_defs):
    xb = xa + int(fieldlength)
    field_defs[i] = (fieldname.strip(), (xa, xb))
    xa = xb
    

wf = open(EXTRACTED_DATA_FILENAME, 'w')
cf = csv.DictWriter(wf, fieldnames=[fd[0] for fd in field_defs])
cf.writeheader()
with open(RAW_DATA_TXT_FILENAME, 'r', encoding='ISO-8859-2') as rf:
    for line in rf:
        cf.writerow({fieldname: line[x1:x2].strip() for fieldname, (x1, x2) in field_defs})
wf.close()



