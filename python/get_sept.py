import os,sys,re,collections
from lxml import html
import requests

base_url = 'http://sacred-texts.com/bib/sep'
top_url = '{}/index.htm'.format(base_url)

page = requests.get(top_url)
tree = html.fromstring(page.content)

books = collections.OrderedDict()
start = False
for x in tree.iter('a'):
    link_text = ''.join(y.text if y.text != None else '' for y in x.iter())
    if not start and link_text == 'Genesis': start = True
    elif not start: continue
    link = x.get('href')
    books[link_text] = '{}/{}'.format(base_url, link)
print(', '.join(books))

chapters = collections.defaultdict(dict)

def getchapters(book):
    book_url = books[book]
    page = requests.get(book_url)
    tree = html.fromstring(page.content)
    chfilter = re.compile(book+' Chapter ([0-9]+)')
    for p in tree.iter('p'):
        for x in p.iter('a'):
            link_text = ''.join(y.text if y.text != None else '' for y in x.iter())
            match = chfilter.match(link_text)
            if match:
                chnum = int(match.group(1))
                link = x.get('href')
                chapters[book][chnum] = '{}/{}'.format(base_url, link)
    print('{}: {} chapters'.format(book, max(x for x in chapters[book])))

for book in books: getchapters(book)

def getchapter(book, chapter):
    url = chapters[book][chapter]
    page = requests.get(url)
    page.encoding = 'utf-8'
    tree = html.fromstring(page.content)
    chtext = ['\n{} {}\n'.format(book, chapter)]
    for x in tree.iter('p'):
        chtext.append(x.text_content())
    return chtext

sf = open('septuagint.txt', 'w')
for book in books:
    sys.stdout.write('writing {} '.format(book))
    sys.stdout.flush()
    for chapter in chapters[book]:
        sys.stdout.write('.')
        sys.stdout.flush()
        sf.write('\n'.join(getchapter(book, chapter)))
    sys.stdout.write('\n')
    sys.stdout.flush()
sf.close()



