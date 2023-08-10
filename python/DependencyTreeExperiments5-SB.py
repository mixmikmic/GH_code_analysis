import pandas as pd
import spacy
import os
import sys
from nltk import Tree
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from internal_displacement.pipeline import SQLArticleInterface

import textacy

nlp = spacy.load('en')

def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
    else:
        return node.orth_

pipeline = SQLArticleInterface("../sql_db.sqlite") #Connecting to pre-populated database.
labels,features = pipeline.get_training_data()

person_reporting_terms = [
    'displaced', 'evacuated', 'forced','flee', 'homeless', 'relief camp',
    'sheltered', 'relocated', 'stranded','stuck','stranded',"killed","dead","died"
]

structure_reporting_terms = [
    'destroyed','damaged','swept','collapsed','flooded','washed'
]

person_reporting_units = ["families","person","people","individuals","locals","villagers","residents","occupants","citizens"]

structure_reporting_units = ["home","house","hut","dwelling","building","shop","business","apartment","flat","residence"]


person_term_lemmas = [t.lemma_ for t in nlp(" ".join(person_reporting_terms))]
structure_term_lemmas = [t.lemma_ for t in nlp(" ".join(structure_reporting_terms))]
person_unit_lemmas = [t.lemma_ for t in nlp(" ".join(person_reporting_units))]
structure_unit_lemmas = [t.lemma_ for t in nlp(" ".join(structure_reporting_units))]

reporting_term_lemmas = person_term_lemmas + structure_term_lemmas
reporting_unit_lemmas = person_unit_lemmas + structure_unit_lemmas

class Report:
    def __init__(self,locations,date_times,event_term,subject_term,quantity,story):
        self.locations = locations
        if date_times:
            self.date_times = [date for date in date_times]
        else:
            self.date_times = []
        self.event_term = event_term
        self.subject_term = subject_term
        self.quantity = quantity
        self.story = story
    
    def display(self):
        print("Location: {}  DateTime: {}  EventTerm: {}  SubjectTerm:  {}  Quantity: {}"
              .format(self.locations,self.date_times,self.event_term,self.subject_term,self.quantity))
        
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

def get_descendents(sentence,root=None):
    """
    Retrieves all tokens that are descended from the specified root token.
    param: root: the root token
    param: sentence: a span from which to retrieve tokens.
    returns: a list of tokens
    """
    if not root:
        root = sentence.root
    return [t for t in sentence if root.is_ancestor_of(t)]

def get_head_descendents(sentence,root=None):
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
    

def get_distance_from_root(token,root):
    """
    Gets the parse tree distance between a token and the sentence root.
    :param token: a token
    :param root: the root token of the sentence
    
    returns: an integer distance
    """
    if token == root:
        return 0
    d = 1
    p = token.head
    while p is not root:
        d+=1
        p = p.head
    return d


def get_common_ancestors(tokens):
    ancestors = [set(t.ancestors) for t in tokens]
    if len(ancestors) == 0:
        return []
    common_ancestors = ancestors[0].intersection(*ancestors)
    return common_ancestors    


def get_distance_between_tokens(token_a,token_b):

    if token_b in token_a.subtree:
        distance = get_distance_from_root(token_b,token_a)
    elif token_a in token_b.subtree:
        distance = get_distance_from_root(token_a,token_b)
    else:
        common_ancestors = get_common_ancestors([token_a,token_b])
        distance = 10000
        for ca in common_ancestors:
            distance_a = get_distance_from_root(ca,token_a)
            distance_b = get_distance_from_root(ca,token_b)
            distance_ab = distance_a + distance_b
            if distance_ab < distance:
                distance = distance_ab
    return distance


def get_closest_contiguous_location_block(entity_list,root_node):
    location_entity_tokens = [[token for token in sentence] for sentence in entity_list]
    token_list =  [item for sublist in location_entity_tokens for item in sublist]
    location_tokens_by_distance = sorted([(token,get_distance_between_tokens(token,root_node)) 
                                          for token in token_list],key= lambda x: x[1])
    closest_location = location_tokens_by_distance[0]
    contiguous_block = [closest_token]
    added_tokens = 1
    while added_tokens > 0:
        contiguous_block_ancestors = [[token for token in token_list if token.is_ancestor_of(toke)] for toke in contiguous_block ]
        contiguous_block_subtrees = [token.subtree for token in contiguous_block]
        contiguous_block_neighbours = contiguous_block_ancestors + contiguous_block_subtrees
        contiguous_block_neighbours = [item for sublist in contiguous_block_neighbours for item in sublist]
        added_tokens = 0
        for toke in token_list:
            if not check_if_collection_contains_token(toke,contiguous_block):
                if toke in contiguous_block_neighbours:
                    added_tokens +=1
                    contiguous_block.append(toke)
    return contiguous_block



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
        if len(block_locations) > 0:
            return [location.text for location in block_locations]
        else:
            return location_entities #If we cannot decide which one is correct, choose them all
                                    #and figure it out at the report merging stage.
    else:
        return []
def extract_dates(sentence,root=None):
    """
    Examines a sentence and identifies if any of its constituent tokens describe a date.
    If a root token is specified, only date tokens below the level of this token in the tree will be examined. 
    If no root is specified, date tokens will be drawn from the entirety of the span.
    Unlike the extract dates function (which returns a list of strings),
    this function returns a list of spacy spans. This is because numerical quantities detected in the 
    branch_search need to be checked to ensure they are not in fact parts of a date.
    
    param: sentence       a span
    param: root           a token
    returns: A list of spacy spans
    """
    if not root:
        root = sentence.root
    descendents = get_head_descendents(sentence,root)
    date_entities = [e for e in nlp(sentence.text).ents if e.label_ == "DATE"]
    if len(date_entities) > 0:
        descendent_date_tokens = []
        for date_ent in date_entities:
            if check_if_entity_contains_token(date_ent,descendents):
                descendent_date_tokens.extend([token for token in date_ent])
        contiguous_token_block = get_contiguous_tokens(descendent_date_tokens)

        block_dates = match_entities_in_block(date_entities,contiguous_token_block)
        return block_dates
    else:
        return None


def basic_number(token):
    if token.text == "dozens":
        return True
    if token.like_num:
        return True
    else:
        return False

def process_sentence_new(sentence, dates_memory, locations_memory, story):
    """
    Extracts the main verbs from a sentence as a starting point
    for report extraction.
    """
    sentence_reports = []
    # Find the verbs
    main_verbs = textacy.spacy_utils.get_main_verbs_of_sent(sentence)
    for v in main_verbs:
        unit_type, verb_lemma = verb_relevance(v)
        if unit_type:
            reports = branch_search_new(v, verb_lemma, unit_type, dates_memory, locations_memory, sentence, story)
            sentence_reports.extend(reports)
    return sentence_reports

def verb_relevance(verb):
    """
    Checks a verb for relevance by:
    1. Comparing to structure term lemmas
    2. Comparing to person term lemmas
    3. Looking for special cases such as 'leave homeless'
    """
    if verb.lemma_ in structure_term_lemmas:
        return structure_unit_lemmas, verb.lemma_
    elif verb.lemma_ in person_term_lemmas:
        return person_unit_lemmas, verb.lemma_
    elif verb.lemma_ == 'leave':
        children = verb.children
        obj_predicate = None
        for child in children:
            if child.dep_ == 'oprd':
                obj_predicate = child
        if obj_predicate:
            if obj_predicate.lemma_ in structure_term_lemmas:
                return structure_unit_lemmas, 'leave ' + obj_predicate.lemma_
            elif obj_predicate.lemma_ in person_term_lemmas:
                return person_unit_lemmas, 'leave ' + obj_predicate.lemma_
    return None, None

def get_quantity_from_phrase(phrase):
    """
    Look for number-like tokens within noun phrase.
    """
    for token in phrase:
        if basic_number(token):
            return token
            
def get_quantity(sentence, unit):
    """
    Split a sentence into noun phrases.
    Search for quantities within each noun phrase.
    If the noun phrase is part of a conjunction, then
    search for quantity within preceding noun phrase
    """
    noun_phrases = list(nlp(sentence.text).noun_chunks)
    # Case one - see if phrase contains the unit
    for i, np in enumerate(noun_phrases):
        if unit.text in np.text:
            if unit.dep_ == 'conj':
                return get_quantity_from_phrase(noun_phrases[i-1])
            else:
                return get_quantity_from_phrase(np)

def get_subjects_and_objects(story, verb):
    """
    Identify subjects and objects for a verb
    Also check if a reporting unit directly precedes
    a verb and is a direct or prepositional object
    """
    verb_objects = textacy.spacy_utils.get_objects_of_verb(verb)
    verb_subjects = textacy.spacy_utils.get_subjects_of_verb(verb)
    verb_objects.extend(verb_subjects)
    #see if unit directly precedes verb
    if verb.i > 0:
        preceding = story[verb.i - 1]
        if preceding.dep_ in ('pobj', 'dobj') and preceding not in verb_objects:
            verb_objects.append(preceding)
    return verb_objects
            
def branch_search_new(verb, verb_lemma, search_type, dates_memory, locations_memory, sentence, story):
    """
    Extract reports based upon an identified verb (reporting term).
    Extract possible locations or use most recent locations
    Extract possible dates or use most recent dates
    Identify reporting unit by looking in objects and subjects of reporting term (verb)
    Identify quantity by looking in noun phrases.
    """
    possible_locations = extract_locations(sentence)
    possible_dates = extract_dates(sentence)
    if not possible_locations:
        possible_locations = locations_memory
    if not possible_dates:
        possible_dates = dates_memory

    reports = []
    quantity = None
    verb_objects = get_subjects_and_objects(story, verb)
    for o in verb_objects:
        if o.lemma_ in search_type:
            # Try and get a number
            quantity = get_quantity(sentence, o)
            report = Report(possible_locations, possible_dates, verb_lemma,
                                    o.lemma_, quantity, story.text)
            reports.append(report)
            #report.display()
    return reports

def process_article_new(story):
    """
    Process a story once sentence at a time
    """
    processed_reports = []
    #if len(story) < 25:
    #    return processed_reports
    story = nlp(story)
    sentences = list(story.sents) # Split into sentences
    dates_memory = None # Keep a running track of the most recent dates found in articles
    locations_memory = None # Keep a running track of the most recent locations found in articles
    for sentence in sentences: # Process sentence
        reports = []
        reports = process_sentence_new(sentence, dates_memory, locations_memory, story)
        current_locations = extract_locations(sentence)
        if current_locations:
            locations_memory = current_locations
        current_dates = extract_dates(sentence)
        if current_dates:
            dates_memory = current_dates
        processed_reports.extend(reports)
    return list(set(processed_reports))

article = "It was early Saturday when a flash flood hit the area and washed away more than 500 houses"
process_article_new(article)

article = "More than fifty homes and shops were destroyed and thousands of acres of farmland flooded."
process_article_new(article)

article = "Quoting an official from the Badakhshan provincial government, Xinhua also said that the foods had damaged or destroyed more than 120 houses in the district."
process_article_new(article)

article = "Mountainous Afghanistan was the worst hit, with 61 people killed and approximately 500 traditional mud-brick homes washed away in more than a dozen villages in Sarobi, a rural district less than an hour from Kabul, officials said."
process_article_new(article)

article = "The June 17 tornado whipped through Essa Township around the supper hour, leaving 100 families homeless while others had to clean up downed trees and debris."
process_article_new(article)

article = "Within hours of the storm, Dowdall had declared a state of emergency and brought in Essa Township emergency departments staff, as well Simcoe County administrators, to assist the 300 people displaced by the storm."
reports = process_article_new(article)

features = pd.DataFrame(features, columns=['content'])
features = features[~features['content'].isin(['', 'retrieval_failed'])]

features['new_reports'] = features['content'].apply(lambda x: process_article_new(x))

features['num_new_reports'] = features['new_reports'].apply(lambda x: len(x))

fail_cases = features[features['num_new_reports'] == 0]

len(fail_cases)

fail_cases.iloc[7]['content']

fail_cases.iloc[8]['content']

fail_cases.iloc[18]['content']



