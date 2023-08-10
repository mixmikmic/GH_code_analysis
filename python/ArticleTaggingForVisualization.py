import pandas as pd
import spacy
import os
import sys
from nltk import Tree
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

nlp = spacy.load('en')

def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
    else:
        return node.orth_

def remove_newline(text):
    ''' Removes new line and &nbsp characters.
    '''
    text = text.replace('\n', ' ')
    text = text.replace('\xa0', ' ')
    return text

test_data = pd.read_csv('../data_extract/article_contents.csv') #Connecting to pre-populated dataset.
test_data['content'] = test_data['content'].apply(lambda x: (remove_newline(str(x))))

person_reporting_terms = [
    'displaced', 'evacuated', 'forced flee', 'homeless', 'relief camp',
    'sheltered', 'relocated', 'stranded','stuck','stranded',"killed","dead","died"
]

structure_reporting_terms = [
    'destroyed','damaged','swept','collapsed','flooded','washed'
]

person_reporting_units = ["families","person","people","individuals","locals","villagers","residents","occupants","citizens", "households"]

structure_reporting_units = ["home","house","hut","dwelling","building","shop","business","apartment","flat","residence"]


person_term_lemmas = [t.lemma_ for t in nlp(" ".join(person_reporting_terms))]
structure_term_lemmas = [t.lemma_ for t in nlp(" ".join(structure_reporting_terms))]
person_unit_lemmas = [t.lemma_ for t in nlp(" ".join(person_reporting_units))]
structure_unit_lemmas = [t.lemma_ for t in nlp(" ".join(structure_reporting_units))]

reporting_term_lemmas = person_term_lemmas + structure_term_lemmas
reporting_unit_lemmas = person_unit_lemmas + structure_unit_lemmas

class Report:
    def __init__(self,locations,date_time,event_term,subject_term,quantity,story):
        self.locations = locations
        self.date_time = date_time
        self.event_term = [t.lemma_ for t in nlp(event_term)][0]
        self.subject_term = subject_term
        self.quantity = quantity
        self.story = story
    
    def display(self):
        print("Location: {}  DateTime: {}  EventTerm: {}  SubjectTerm:  {}  Quantity: {}"
              .format(self.locations,self.date_time,self.event_term,self.subject_term,self.quantity))
        
    def show_story_tree(self):
        self.display()
        for sentence in nlp(self.story).sents:
            for token in sentence:
                if token.lemma_ == self.event_term:
                    return to_nltk_tree(sentence.root)

def test_token_equality(token_a,token_b):
    if token_a.text == token_b.text:
        return True
    else:
        return False
    
def check_if_collection_contains_token(token,collection):
    if any([test_token_equality(token,t) for t in collection]):
        return True
    else:
        return False

def check_if_entity_contains_token(tokens,entity):
    """
    Function to test if a given entity contains at least one of a list of tokens.
    param: tokens: A list of tokens
    param: entity: A span
    
    returns: Boolean
    """
    tokens_ = [t.text for t in tokens]
    ret = False
    for token in entity:
        if token.text in tokens_:
            return True
    return False

def extract_locations(sentence,root=None):
    """
    Examines a sentence and identifies if any of its constituent tokens describe a location.
    If a root token is specified, only location tokens below the level of this token in the tree will be examined. 
    If no root is specified, location tokens will be drawn from the entirety of the span.
    param: sentence       a span
    param: root           a token
    returns: A list of strings, or None
    """
    if not root:
        root = sentence.root
    descendents = get_descendents(sentence,root)
    location_entities = [e for e in nlp(sentence.text).ents if e.label_ == "GPE"]
    if len(location_entities) > 0:
        descendent_location_tokens = []
        for location_ent in location_entities:
            if check_if_entity_contains_token(location_ent,descendents):
                descendent_location_tokens.extend([token for token in location_ent])
        contiguous_token_block = get_contiguous_tokens(descendent_location_tokens)

        block_locations = match_entities_in_block(location_entities,contiguous_token_block)
        return [location.text for location in block_locations]
    else:
        return None
    
def extract_dates(sentence,root=None):
    """
    Examines a sentence and identifies if any of its constituent tokens describe a date.
    If a root token is specified, only date tokens below the level of this token in the tree will be examined. 
    If no root is specified, date tokens will be drawn from the entirety of the span.
    param: sentence       a span
    param: root           a token
    returns: A list of strings, or None
    """
    if not root:
        root = sentence.root
    descendents = get_descendents(sentence,root)
    date_entities = [e for e in nlp(sentence.text).ents if e.label_ == "DATE"]
    if len(date_entities) > 0:
        descendent_date_tokens = []
        for date_ent in date_entities:
            if check_if_entity_contains_token(date_ent,descendents):
                descendent_date_tokens.extend([token for token in date_ent])
        contiguous_token_block = get_contiguous_tokens(descendent_date_tokens)

        block_dates = match_entities_in_block(date_entities,contiguous_token_block)
        return [location.text for location in block_dates]
    else:
        return None
    
def get_contiguous_tokens(token_list):
    common_ancestor_tokens = get_common_ancestors(token_list)
    highest_contiguous_block = []
    for toke in token_list:
        if check_if_collection_contains_token(toke.head,common_ancestor_tokens):
            highest_contiguous_block.append(toke)
    added_tokens = 1
    while added_tokens > 0:
        added_tokens = 0
        for toke in token_list:
            if check_if_collection_contains_token(toke.head,highest_contiguous_block):
                if not check_if_collection_contains_token(toke,highest_contiguous_block):
                    highest_contiguous_block.append(toke)
                    added_tokens +=1
    return highest_contiguous_block

def match_entities_in_block(entities,token_block):
    matched = []
    text_block = [t.text for t in token_block] #For some reason comparing identity on tokens does not always work.
    for e in entities:
        et = [t.text for t in e]
        et_in_b = [t for t in et if t in text_block]
        if len(et_in_b) == len(et):
            matched.append(e)
    return matched

def get_common_ancestors(tokens):
    ancestors = [set(t.ancestors) for t in tokens]
    if len(ancestors) == 0:
        return []
    common_ancestors = ancestors[0].intersection(*ancestors)
    return common_ancestors


def get_descendents(sentence,root=None):
    """
    Retrieves all tokens that are descended from the head of the specified root token.
    param: root: the root token
    param: sentence: a span from which to retrieve tokens.
    returns: a list of tokens
    """
    if not root:
        root = sentence.root
    else:
        root = root.head
    return [t for t in sentence if root.is_ancestor_of(t)]
    
def get_all_descendent_tokens(token):
    """
    Returns a list of all descendents of the specified token.
    """
    children_accum = []
    for child in token.children:
        children_accum.append(child)
        grandchildren = get_all_descendent_tokens(child)
        children_accum.extend(grandchildren)
    return children_accum

def process_branch(token):
    '''Examines a branch (defined as token and all of its children)
    to see if any tokens are number-like and / or reporting units
    If a reporting_unit is found, returns the identified unit and any
    identified numbers
    param: token       a token
    return: reporting_unit, number or None, None
    '''
    children = [token] + get_all_descendent_tokens(token)
    reporting_unit, number = None, None
    for child in children:
        if child.like_num:
            number = child.text
        elif child.lemma_ in reporting_unit_lemmas:
            reporting_unit = child.text
    return reporting_unit, number

def process_article(story):
    '''Process an article by splitting it into sentences and
    calling process_sentence for each sentence
    Keep a running track of identified dates and locations that
    can be used as default values for reports that have no date
    or location
    param: story       string
    return: list of reports
    '''
    processed_reports = []
    sentences = list(nlp(story).sents) # Split into sentences
    last_date = None # Keep a running track of the most recent date found in articles
    last_location = None # Keep a running track of the most recent location found in articles
    for sentence in sentences: # Process sentence
        report = process_sentence(sentence, story)
        if report:
            if report.date_time:
                last_date = report.date_time
            else:
                report.date_time = last_date
            if report.locations:
                last_location = report.locations
            else:
                report.locations = last_location
            processed_reports.append(report)
    return processed_reports

def process_sentence(sentence, story):
    '''Process a sentence to try and find any reports contained
    within it.
    First try and find a reporting_term; if it exists identify any
    locations and dates.
    Finally, look within all branches below the reporting_term to
    try and identify a relevant reporting unit and number.
    If a minimum of a reporting_term and reporting_unit exist, 
    then create a report.
    param: sentence Spacy sentence
    return: report
    '''
    for token in sentence:
        if token.lemma_ in reporting_term_lemmas:
            term_token = token
            possible_locations = extract_locations(sentence,token)
            possible_dates = extract_dates(sentence,token)
            reporting_term = term_token.text 
            children = term_token.children
            for child in children:
                reporting_unit, number = process_branch(child)
                if reporting_unit:
                    report = Report(possible_locations,possible_dates,reporting_term,reporting_unit,number,story)
                    return report

article = test_data.iloc[0]['content']
print("=============Story================")
print(article)
print("=============Reports================")
reports = process_article(article)
for report in reports:
    report.display()

def tag_sentence(sentence):
    start_tag = '<mark data-entity="report">'
    end_tag = '</mark>'
    return start_tag + sentence + end_tag

def process_article(story):
    processed_reports = []
    tagged_article = []
    sentences = list(nlp(story).sents) # Split into sentences
    last_date = None # Keep a running track of the most recent date found in articles
    last_location = None # Keep a running track of the most recent location found in articles
    for sentence in sentences: # Process sentence
        report = process_sentence(sentence, story)
        if report:
            tagged_article.append(tag_sentence(sentence.text))
            if report.date_time:
                last_date = report.date_time
            else:
                report.date_time = last_date
            if report.locations:
                last_location = report.locations
            else:
                report.locations = last_location
            processed_reports.append(report)
        else:
            tagged_article.append(sentence.text)
    return processed_reports, tagged_article # If implemented, update Article with tagged version

article = test_data.iloc[0]['content']
print("=============Tagged Article================")
reports, tagged_article = process_article(article)
print(tagged_article)

def tag_token(token, data_type):
    start_tag = '<mark data-entity="{}">'.format(data_type)
    end_tag = '</mark>'
    return start_tag + token + end_tag

def apply_tags(story, tag_set):
    if tag_set['reporting_term']:
        story[tag_set['reporting_term']] = tag_token(story[tag_set['reporting_term']], 'reporting_term')
    if tag_set['reporting_unit']:
        story[tag_set['reporting_unit']] = tag_token(story[tag_set['reporting_unit']], 'reporting_unit')
    if tag_set['number']:
        story[tag_set['number']] = tag_token(story[tag_set['number']], 'number')
    if tag_set['dates']:
        for idx in tag_set['dates']:
            story[idx] = tag_token(story[idx], 'date')
    if tag_set['locations']:
        for idx in tag_set['locations']:
            story[idx] = tag_token(story[idx], 'location')
    return story

def apply_tags_to_article(article, tag_indices):
    if len(tag_indices) > 0:
        story = [tag.text for tag in article]
        for tag_set in tag_indices:
            story = apply_tags(story, tag_set)
    return " ".join(story) + "."
    
def extract_locations(sentence,root=None):
    if not root:
        root = sentence.root
    descendents = get_descendents(sentence,root)
    location_entities = [e for e in nlp(sentence.text).ents if e.label_ == "GPE"]
    if len(location_entities) > 0:
        descendent_location_tokens = []
        for location_ent in location_entities:
            if check_if_entity_contains_token(location_ent,descendents):
                descendent_location_tokens.extend([token for token in location_ent])
        contiguous_token_block = get_contiguous_tokens(descendent_location_tokens)

        block_locations = match_entities_in_block(location_entities,contiguous_token_block)
        return [location.text for location in block_locations], None
    else:
        return None, None
    
def extract_dates(sentence,root=None):
    if not root:
        root = sentence.root
    descendents = get_descendents(sentence,root)
    date_entities = [e for e in nlp(sentence.text).ents if e.label_ == "DATE"]
    if len(date_entities) > 0:
        descendent_date_tokens = []
        for date_ent in date_entities:
            if check_if_entity_contains_token(date_ent,descendents):
                descendent_date_tokens.extend([token for token in date_ent])
        contiguous_token_block = get_contiguous_tokens(descendent_date_tokens)

        block_dates = match_entities_in_block(date_entities,contiguous_token_block)
        return [location.text for location in block_dates], None
    else:
        return None, None

def process_branch(token):
    children = [token] + get_all_descendent_tokens(token)
    reporting_unit, number = None, None
    reporting_unit_idx, number_idx = None, None
    for child in children:
        if child.like_num:
            number = child.text
            number_idx = child.i
        elif child.lemma_ in reporting_unit_lemmas:
            reporting_unit = child.text
            reporting_unit_idx = child.i
    return reporting_unit, number, (reporting_unit_idx, number_idx)

def process_sentence(sentence, story):
    tag_indices = {
        'dates': None, 'locations': None, 'reporting_term': None,
        'reporting_unit': None, 'number': None }
    for token in sentence:
        if token.lemma_ in reporting_term_lemmas:
            tag_indices['reporting_term'] = token.i
            term_token = token
            possible_locations, locations_indices = extract_locations(sentence,token)
            tag_indices['locations'] = locations_indices
            possible_dates, dates_indices = extract_dates(sentence,token)
            tag_indices['dates'] = dates_indices
            reporting_term = term_token.text 
            children = term_token.children
            for child in children:
                reporting_unit, number, indices = process_branch(child)
                if reporting_unit:
                    tag_indices['reporting_unit'] = indices[0]
                    tag_indices['number'] = indices[1]
                    report = Report(possible_locations,possible_dates,reporting_term,reporting_unit,number,story)
                    return report, tag_indices
    return None, None
                
def process_article(story):
    processed_reports = []
    article_report_indices = []
    story = nlp(story)
    sentences = list(story.sents) # Split into sentences
    last_date = None # Keep a running track of the most recent date found in articles
    last_location = None # Keep a running track of the most recent location found in articles
    for sentence in sentences: # Process sentence
        report, report_indices = process_sentence(sentence, story)
        if report:
            article_report_indices.append(report_indices)
            if report.date_time:
                last_date = report.date_time
            else:
                report.date_time = last_date
            if report.locations:
                last_location = report.locations
            else:
                report.locations = last_location
            processed_reports.append(report)
    tagged_article = apply_tags_to_article(story, article_report_indices)
    return processed_reports, tagged_article # If implemented, update Article with tagged version

article = test_data.iloc[0]['content']
print("=============Tagged Article================")
reports, tagged_article = process_article(article)
print(tagged_article)

def tag_text(text, spans):
    text_blocks = []
    text_start_point = 0
    for span in spans:
            text_blocks.append(text[text_start_point : span['start']])

            tagged_text = '<mark data-entity="{}">'.format(span['type'].lower())
            tagged_text += text[span['start'] : span['end']]
            tagged_text += '</mark>'
            text_blocks.append(tagged_text)
            text_start_point = span['end']
    text_blocks.append(text[text_start_point : ])
    return("".join(text_blocks))


def extract_locations(sentence,root=None):
    sentence_start = sentence.start_char
    if not root:
        root = sentence.root
    descendents = get_descendents(sentence,root)
    location_entities = [e for e in nlp(sentence.text).ents if e.label_ == "GPE"]
    if len(location_entities) > 0:
        descendent_location_tokens = []
        for location_ent in location_entities:
            if check_if_entity_contains_token(location_ent,descendents):
                descendent_location_tokens.extend([token for token in location_ent])
        contiguous_token_block = get_contiguous_tokens(descendent_location_tokens)

        block_locations = match_entities_in_block(location_entities,contiguous_token_block)
        locations_spans = []
        for location in block_locations:
            loc_start = location.start_char + sentence_start
            loc_end = location.end_char + sentence_start
            span = {'start': loc_start, 'end': loc_end, 'type': 'LOC'}
            locations_spans.append(span)
        return [location.text for location in block_locations], locations_spans
    else:
        return None, None

    
def extract_dates(sentence,root=None):
    sentence_start = sentence.start_char
    if not root:
        root = sentence.root
    descendents = get_descendents(sentence,root)
    date_entities = [e for e in nlp(sentence.text).ents if e.label_ == "DATE"]
    if len(date_entities) > 0:
        descendent_date_tokens = []
        for date_ent in date_entities:
            if check_if_entity_contains_token(date_ent,descendents):
                descendent_date_tokens.extend([token for token in date_ent])
        contiguous_token_block = get_contiguous_tokens(descendent_date_tokens)

        block_dates = match_entities_in_block(date_entities,contiguous_token_block)
        dates_spans = []
        for location in block_dates:
            loc_start = location.start_char + sentence_start
            loc_end = location.end_char + sentence_start
            span = {'start': loc_start, 'end': loc_end, 'type': 'DATE'}
            dates_spans.append(span)
        return [location.text for location in block_dates], dates_spans
    else:
        return None, None

    
def process_branch(token):
    children = [token] + get_all_descendent_tokens(token)
    reporting_unit, number = None, None
    spans = []
    for child in children:
        if child.like_num:
            number = child.text
            span = {'start': child.idx, 'end': len(child) + child.idx, 'type': 'NUM'}
            spans.append(span)
        elif child.lemma_ in reporting_unit_lemmas:
            reporting_unit = child.text
            span = {'start': child.idx, 'end': len(child) + child.idx, 'type': 'UNIT'}
            spans.append(span)
    return reporting_unit, number, spans


def process_sentence(sentence, story):
    spans = []
    for token in sentence:
        if token.lemma_ in reporting_term_lemmas:
            term_span = {'start': token.idx, 'end': len(token) + token.idx, 'type': 'TERM'}
            spans.append(term_span)
            term_token = token
            possible_locations, locations_spans = extract_locations(sentence,token)
            
            if locations_spans:
                spans.extend(locations_spans)
            
            possible_dates, dates_spans = extract_dates(sentence,token)
            if dates_spans:
                spans.extend(dates_spans)
            reporting_term = term_token.text 
            children = term_token.children
            for child in children:
                reporting_unit, number, child_spans = process_branch(child)
                if reporting_unit:
                    spans.extend(child_spans)
                    report = Report(possible_locations,possible_dates,reporting_term,reporting_unit,number,story)
                    return report, spans
    return None, None


def process_article(story):
    processed_reports = []
    spans = []
    story = nlp(story)
    sentences = list(story.sents) # Split into sentences
    last_date = None # Keep a running track of the most recent date found in articles
    last_location = None # Keep a running track of the most recent location found in articles
    for sentence in sentences: # Process sentence
        report, sentence_spans = process_sentence(sentence, story)
        if report:
            spans.extend(sentence_spans)
            if report.date_time:
                last_date = report.date_time
            else:
                report.date_time = last_date
            if report.locations:
                last_location = report.locations
            else:
                report.locations = last_location
            processed_reports.append(report)
    return processed_reports, spans  # If implemented, update Article with tagged version

article = test_data.iloc[0]['content']
reports, spans = process_article(article)
print("=============Reports================")
for report in reports:
    report.display()
print("==============Span=================")
for span in spans:
    print(span)
print("==============Tagged Article=================")
print(tag_text(article, spans))

article = test_data.iloc[1]['content']
reports, spans = process_article(article)
print("=============Reports================")
for report in reports:
    report.display()
print("==============Span=================")
for span in spans:
    print(span)
print("==============Tagged Article=================")
print(tag_text(article, spans))



