# Import base packages
from bs4 import BeautifulSoup
import os, glob, sys, re
import codecs
import pandas as pd

# Let's explore one content page
html = '../sample/html/1.1.1.1.1.1.html'
page = codecs.open(html, 'r', encoding='utf-8')

# Extract page contents
soup = BeautifulSoup(page.read(), 'html.parser')

# The navigation path titles are included in {'class': 'breadcrumb'}
titles = soup.find('ol', {'class': 'breadcrumb'}).findAll('a')
print('HTML source: %s' % titles)

# Extract title texts
for title in titles:
    print(title.get('title'))

# Extract chapter, section and subsection titles
chapter_title    = ' - '.join([x.get('title').split('-')[1].strip() for x in titles[2:4]])
section_title    = ' - '.join([x.get('title').split('-')[1].strip() for x in titles[4:]])
subsection_title = soup.find(id='page-title').text.split('-')[1].strip()

print('Chapter title   : %s' % chapter_title)
print('Section title   : %s' % section_title)
print('Subsection title: %s' % subsection_title)

# Strip non-ascii characters that break the overlap check
def strip_non_ascii(s):
    s = (c for c in s if 0 < ord(c) < 255)
    s = ''.join(s)
    return s

# Clean text: remove newlines, compact spaces, strip non_ascii, etc.
def clean_text(text, lowercase=False, nopunct=False):
    # Convert to lowercase
    if lowercase:
        text = text.lower()

    # Remove punctuation
    if nopunct:
        puncts = string.punctuation
        for c in puncts:
            text = text.replace(c, ' ')

    # Strip non-ascii characters
    text = strip_non_ascii(text)
    
    # Remove newlines - Compact and strip whitespaces
    text = re.sub('[\r\n]+', ' ', text)
    text = re.sub('\s+', ' ', text)
    return text.strip()

# Get content from all subsections in page at once
def get_content_all(soup):
    section = soup.find("div", { "class" : "section" })
    section_text = ''
    
    # If page is empty, return
    if section == None:
        return section_text
    
    divs = section.findAll('div')
    for div in divs:
        # Do not include'sourceCredit' or 'section inline'
        if div.get('class')[0] != 'sourceCredit':
            # Fix a formatting issue causing some text to be collated when extracted
            for sp in div.findAll("span", { "class" : "chapeau" }):
                sp.replaceWith('<sp>' + sp.text)
            section_text += div.text.replace('<sp>', ' ')
    
    # Clean text, do not convert to lowercase or remove punctuation (default)
    section_text = clean_text(section_text, lowercase=False, nopunct=False)
    
    return section_text

content = get_content_all(soup)
print('Content text:\n%s' % content)

from rake import *

# Extract keyphrases using RAKE algorithm. Limit results by minimum score.
def get_keyphrases_rake(text, stoplist_path=None, min_score=0):
    if stoplist_path == None:
        stoplist_path = 'SmartStoplist.txt'

    rake = Rake(stoplist_path)
    keywords = rake.run(text)
    phrases = []
    for keyword in keywords:
        score = keyword[1]
        if score >= min_score:
            phrases.append(keyword)

    return phrases

stoplist_file = 'SmartStoplist_extended.txt'
keyphrases = get_keyphrases_rake(content, stoplist_path=stoplist_file, min_score=3)

print('Number of keyphrases = %d' % len(keyphrases))
for keyphrase in keyphrases:
    print('%s -> %f' % keyphrase)

import pke

def load_stop_words(stoplist_path):
    stop_words = []
    for line in open(stoplist_path):
        if line.strip()[0:1] != "#":
            for word in line.split():
                stop_words.append(word)
    return stop_words

def get_keyphrases_pke(text, stoplist_path=None, postags=None):
    if stoplist_path == None:
        stoplist_path = 'SmartStoplist.txt'
    stoplist = load_stop_words(stoplist_path)

    if postags == None:
        postags = ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS', 'VBN', 'VBD']

    # PKE expects an input file. Save text to temporary file to proceed.
    infile = 'tmp_%d.txt' % (os.getpid())
    f = open(infile, 'w')
    print >>f, text.encode('utf8')
    f.close()

    # Run keyphrase extractor (using TOPICRANK algorithm)
    try:
        extractor = pke.TopicRank(input_file=infile, language='english')
        extractor.read_document(format='raw', stemmer=None)
        extractor.candidate_selection(stoplist=stoplist, pos=postags)
        extractor.candidate_weighting(threshold=0.25, method='average')
        phrases = extractor.get_n_best(300, redundancy_removal=True)
    except:
        phrases = []

    # (Optional) Keep unique keywords only
    #phrases = ' '.join(p for p in set(phrases.split()))
    os.remove(infile)
    return phrases

stoplist_file = 'SmartStoplist_extended.txt'
custom_tags = ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS', 'VBN', 'VBD', 'VBG']
keyphrases = get_keyphrases_pke(content, stoplist_path=stoplist_file, postags=custom_tags)

print('Number of keyphrases = %d' % len(keyphrases))
for keyphrase in keyphrases:
    print('%s -> %f' % keyphrase)

# Get header and content from each subsection separately
def get_content_subsections(soup):
    subs = []
    sections = soup.findAll('div', {'class': 'subsection indent2 firstIndent-2'})
    
    if len(sections) > 0:
        for div in sections:
            sub_header = div.find('span', {'class': 'heading bold'})
            # Check if subsection has a valid title
            if sub_header != None:
                sub_title  = clean_text(sub_header.text)
            else:
                sub_title  = ''
            # Fix a formatting issue causing some text to be collated when extracted
            for sp in div.findAll("span", { "class" : "chapeau" }):
                sp.replaceWith('<sp>' + sp.text)
            sub_text = div.text.replace('<sp>', ' ')
            # Clean text, do not convert to lowercase or remove punctuation (default)
            sub_text = clean_text(sub_text, lowercase=False, nopunct=False)
            subs.append((sub_title, sub_text))
    else:
        # If page does not contain subsections, parse it as one
        sub_text = get_content_all(soup)
        # Check if page is empty
        if sub_text != '':
            subs.append(('', sub_text))

    return subs

subsections = get_content_subsections(soup)

print('Found %d subsections' % len(subsections))
for i, subsection in enumerate(subsections):
    print('Subsection# %d: %s' % (i, subsection[0]))
    print('%s\n' % subsection[1])

stoplist_file = 'SmartStoplist_extended.txt'
sub_ind = 0
sub_content = subsections[sub_ind][1]
keyphrases = get_keyphrases_rake(sub_content, stoplist_path=stoplist_file, min_score=1)

print('Number of keyphrases = %d' % len(keyphrases))
for keyphrase in keyphrases:
    print('%s -> %f' % keyphrase)
    
# Combined list of keyphrases to be used for indexing
all_phrases = ', '.join(p[0] for p in keyphrases)
print('\nKeyphrases list: %s' % all_phrases)

def parse_contents(hfile, mode='full_page', stoplist_path=None, min_score=1):
    global df
    infile  = os.path.basename(hfile)
    print('Processing %s' % infile)
    
    # Parse and extract title and sections of interest
    page = codecs.open(hfile, 'r', encoding='utf-8')
    soup  = BeautifulSoup(page.read(), 'html.parser')
    
    # The navigation path titles are included in {'class': 'breadcrumb'}
    titles = soup.find('ol', {'class': 'breadcrumb'}).findAll('a')
    
    # Extract chapter, section and subsection titles
    # Check if chapter title is valid - Handle exception cases
    try:
        chapter_title    = ' - '.join([x.get('title').split('-')[1].strip() for x in titles[2:4]])
    except:
        chapter_title    = ' - '.join([x.get('title').strip() for x in titles[2:4]])
       
    # Check if section title is valid - Handle exception cases
    try:
        section_title    = ' - '.join([x.get('title').split('-')[1].strip() for x in titles[4:]])
    except:
        section_title    = ' - '.join([x.get('title').strip() for x in titles[4:]])

    # Use page title as the base subsection title
    subsection_title = soup.find(id='page-title').text.split('-')[1].strip()
     
    # Option #1 - Extract all page content as one document
    if mode == 'full_page':
        page_text = get_content_all(soup)
        phrases  = get_keyphrases_rake(page_text, stoplist_path=stoplist_file, min_score=min_score)
        phrases  = ', '.join(p[0] for p in phrases)
        df = df.append({'File'           : infile, 
                        'ChapterTitle'   : chapter_title.replace('\r', ''),
                        'SectionTitle'   : section_title.replace('\r', ''),
                        'SubsectionTitle': subsection_title.replace('\r', ''),
                        'SubsectionText' : page_text.replace('\r', ''),
                        'Keywords'       : phrases.replace('\r', '')},
                        ignore_index=True)        
    
    # Option #2 - Extract header and content from each subsection separately
    elif mode == 'split_page':
        subsections = get_content_subsections(soup)
        for i, subsection in enumerate(subsections):
            # append subsection header to main subsection_title
            sub_title = subsection_title
            if subsection[0] != '':
                sub_title = sub_title + ' - ' + subsection[0]
            sub_text = subsection[1]
            phrases  = get_keyphrases_rake(sub_text, stoplist_path=stoplist_file, min_score=min_score)
            phrases  = ', '.join(p[0] for p in phrases)
            df = df.append({'File'           : infile, 
                            'ChapterTitle'   : chapter_title.replace('\r', ''),
                            'SectionTitle'   : section_title.replace('\r', ''),
                            'SubsectionTitle': sub_title.replace('\r', ''),
                            'SubsectionText' : sub_text.replace('\r', ''),
                            'Keywords'       : phrases.replace('\r', '')},
                            ignore_index=True)        
    else:
        print('Invalid parsing mode %s ... Valid options: full_page or split_page')

    print('Finished processing %s ...' % infile)
    return

INDIR  = '../sample/html'
OUTDIR = '../sample'

# Select parsing option: Option #1 (FULL_PAGE), Option #2 (SPLIT_PAGE), or both
FULL_PAGE  = False
SPLIT_PAGE = True

if not os.path.exists(OUTDIR):
  os.makedirs(OUTDIR)

# Dataframe to keep all extracted content fields
df = pd.DataFrame(columns = ['File', 'ChapterTitle', 'SectionTitle', 'SubsectionTitle',
                                     'SubsectionText', 'Keywords'])
    
# Set custom stopwords list, if needed
stoplist_file = 'SmartStoplist_extended.txt'

# Process all content pages
for infile in glob.glob(INDIR + '/*.html'):
    if FULL_PAGE:
        parse_contents(infile, mode='full_page',  stoplist_path=stoplist_file, min_score=3)
    if SPLIT_PAGE:
        parse_contents(infile, mode='split_page', stoplist_path=stoplist_file, min_score=1)

# Save extracted content for indexing in step #2
#outfile = OUTDIR + '/parsed_content.tsv'
#df.to_csv(outfile, sep='\t', index_label='Index', encoding='utf-8')    
outxlsx = OUTDIR + '/parsed_content.xlsx'
df.to_excel(outxlsx, index_label='Index', encoding='utf-8') 

df.head(5)



