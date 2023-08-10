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
import re
import hashlib

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
    'sheltered', 'relocated', 'stranded','stuck','stranded',"killed","dead","died","drown"
]

structure_reporting_terms = [
    'destroyed','damaged','swept','collapsed','flooded','washed', 'inundated', 'evacuate'
]

person_reporting_units = ["families","person","people","individuals","locals","villagers","residents","occupants","citizens", "households"]

structure_reporting_units = ["home","house","hut","dwelling","building","shop","business","apartment","flat","residence"]


person_term_lemmas = [t.lemma_ for t in nlp(" ".join(person_reporting_terms))]
structure_term_lemmas = [t.lemma_ for t in nlp(" ".join(structure_reporting_terms))]
person_unit_lemmas = [t.lemma_ for t in nlp(" ".join(person_reporting_units))]
structure_unit_lemmas = [t.lemma_ for t in nlp(" ".join(structure_reporting_units))]

reporting_term_lemmas = person_term_lemmas + structure_term_lemmas
reporting_unit_lemmas = person_unit_lemmas + structure_unit_lemmas

relevant_article_terms = ['Rainstorm', 'hurricane', 'tornado', 'rain', 'storm', 'earthquake']
relevant_article_lemmas = [t.lemma_ for t in nlp(" ".join(relevant_article_terms))]

class Report:
    def __init__(self,locations,date_times,event_term,subject_term,quantity,story):
        self.locations = locations
        if date_times:
            self.date_times = date_times
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
                
    def report_hash(self):
        report_string = "Location: {}  DateTime: {}  EventTerm: {}  SubjectTerm:  {}  Quantity: {}".format(self.locations,self.date_times,self.event_term,self.subject_term,self.quantity)
        hash1 = hashlib.md5(report_string.encode('utf-8')).hexdigest()
        return hash1
    
    def to_json(self):
        d = {}
        d['Location'] = self.locations
        d['DateTime'] = self.date_times
        d['EventTerm'] = self.event_term
        d['SubjectTerm'] = self.subject_term
        d['Quantity'] = self.quantity
        return d

def test_token_equality(token_a,token_b):
    if token_a.i == token_b.i:
        return True
    else:
        return False
    
def check_if_collection_contains_token(token,collection):
    for c in collection:
        if test_token_equality(token,c):
            return True
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
            return [location.text for location in location_entities] #If we cannot decide which one is correct, choose them all
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
        return [date.text for date in block_dates]
    else:
        return None


def basic_number(token):
    if token.text in ("dozens", "hundreds", "thousands"):
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
        unit_type, verb_lemma = verb_relevance(v, story)
        if unit_type:
            reports = branch_search_new(v, verb_lemma, unit_type, dates_memory, locations_memory, sentence, story)
            sentence_reports.extend(reports)
    return sentence_reports

def article_relevance(article):
    for token in article:
        if token.lemma_ in relevant_article_lemmas:
            return True

def verb_relevance(verb, article):
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
            if child.dep_ in ('oprd', 'dobj'):
                obj_predicate = child
        if obj_predicate:
            if obj_predicate.lemma_ in structure_term_lemmas:
                return structure_unit_lemmas, 'leave ' + obj_predicate.lemma_
            elif obj_predicate.lemma_ in person_term_lemmas:
                return person_unit_lemmas, 'leave ' + obj_predicate.lemma_
    elif verb.lemma_ == 'affect' and article_relevance(article):
        return structure_unit_lemmas + person_unit_lemmas , verb.lemma_
    elif verb.lemma_ in ('fear', 'assume'):
        verb_objects = textacy.spacy_utils.get_objects_of_verb(verb)
        if verb_objects:
            verb_object = verb_objects[0]
            if verb_object.lemma_ in person_term_lemmas:
                return person_unit_lemmas, verb.lemma_ + " " + verb_object.text
            elif verb_object.lemma_ in structure_term_lemmas:
                return structure_unit_lemmas, verb.lemma_ + " " + verb_object.text
        
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
        if check_if_collection_contains_token(unit,np):
            if unit.dep_ == 'conj':
                return get_quantity_from_phrase(noun_phrases[i-1])
            else:
                return get_quantity_from_phrase(np)
    #Case two - get any numeric child of the unit noun.
    for child in unit.children:
        if basic_number(child):
            return child
    

def simple_subjects_and_objects(verb):
    verb_objects = textacy.spacy_utils.get_objects_of_verb(verb)
    verb_subjects = textacy.spacy_utils.get_subjects_of_verb(verb)
    verb_objects.extend(verb_subjects)
    return verb_objects


def nouns_from_relative_clause(sentence, verb):
    possible_clauses = list(textacy.extract.pos_regex_matches(sentence, r'<NOUN>+<VERB>'))
    for clause in possible_clauses:
        if verb in clause:
            for token in clause:
                if token.tag_ == 'NNS':
                    return token

            
def get_subjects_and_objects(story, sentence, verb):
    """
    Identify subjects and objects for a verb
    Also check if a reporting unit directly precedes
    a verb and is a direct or prepositional object
    """
    # Get simple or standard subjects and objects
    verb_objects = simple_subjects_and_objects(verb)
    # Special Cases

    #see if unit directly precedes verb
    if verb.i > 0:
        preceding = story[verb.i - 1]
        if preceding.dep_ in ('pobj', 'dobj') and preceding not in verb_objects:
            verb_objects.append(preceding)

    # See if verb is part of a conjunction
    if verb.dep_ == 'conj':
        lefts = list(verb.lefts)
        if len(lefts) > 0:
            for token in lefts:
                if token.dep_ in ('nsubj', 'nsubjpass'):
                    verb_objects.append(token)
        else:            
            ancestors = verb.ancestors
            for anc in ancestors:
                verb_objects.extend(simple_subjects_and_objects(anc))
            
    # Look for 'pobj' in sentence
    if verb.dep_ == 'ROOT':
        for token in sentence:
            if token.dep_ == 'pobj':
                verb_objects.append(token)
                
    # Look for nouns in relative clauses
    if verb.dep_ == 'relcl':
        relcl_noun = nouns_from_relative_clause(sentence, verb)
        if relcl_noun:
            verb_objects.append(relcl_noun)
        
    
    return list(set(verb_objects))


def test_noun_conj(sentence, noun):
    possible_conjs = list(textacy.extract.pos_regex_matches(sentence, r'<NOUN><CONJ><NOUN>'))
    for conj in possible_conjs:
        if noun in conj:
            return conj

            
def branch_search_new(verb, verb_lemma, search_type, dates_memory, locations_memory, sentence, story):
    """
    Extract reports based upon an identified verb (reporting term).
    Extract possible locations or use most recent locations
    Extract possible dates or use most recent dates
    Identify reporting unit by looking in objects and subjects of reporting term (verb)
    Identify quantity by looking in noun phrases.
    """
    possible_locations = extract_locations(sentence,verb)
    possible_dates = extract_dates(sentence)
    if not possible_locations:
        possible_locations = locations_memory
    if not possible_dates:
        possible_dates = dates_memory
    reports = []
    quantity = None
    verb_objects = get_subjects_and_objects(story, sentence, verb)
    #If there are multiple possible nouns and it is unclear which is the correct one
    #choose the one with the fewest descendents. A verb object with many descendents is more likely to 
    #have its own verb as a descendent.
    verb_descendent_counts = [(v,len(list(v.subtree))) for v in verb_objects]
    verb_objects = [x[0] for x in sorted(verb_descendent_counts,key = lambda x: x[1])]
    for o in verb_objects:
        if basic_number(o) and o.i == (verb.i - 1):
            quantity = o
            if search_type == structure_term_lemmas:
                unit = 'house'
            else:
                unit = 'person'
            report = Report(possible_locations, possible_dates, verb_lemma,
                                    unit, quantity, story.text)
            #report.display()
            reports.append(report)
            break
        elif o.lemma_ in search_type:
            reporting_unit = o.lemma_
            noun_conj = test_noun_conj(sentence, o)
            if noun_conj:
                reporting_unit = noun_conj
            # Try and get a number
            quantity = get_quantity(sentence, o)
            report = Report(possible_locations, possible_dates, verb_lemma,
                                    reporting_unit, quantity, story.text)
            reports.append(report)
            #report.display()
            break
    return reports

def cleanup(text):
    text = re.sub(r'([a-zA-Z0-9])(IMPACT)', r'\1. \2', text)
    text = re.sub(r'([a-zA-Z0-9])(RESPONSE)', r'\1. \2', text)
    text = re.sub(r'(IMPACT)([a-zA-Z0-9])', r'\1. \2', text)
    text = re.sub(r'(RESPONSE)([a-zA-Z0-9])', r'\1. \2', text)
    text = re.sub(r'([a-zA-Z])(\d)', r'\1. \2', text)
    return text

def process_article_new(story):
    """
    Process a story once sentence at a time
    """
    story = cleanup(story)
    processed_reports = []
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

def check_language(text):
    try:
        lang = textacy.text_utils.detect_language(text)
        return lang
    except ValueError:
        return 'na'

def compare_reports(reports1, reports2):
    report_hashes_1 = [r.report_hash() for r in reports1]
    report_hashes_2 = [r.report_hash() for r in reports2]
    equal_length = len(reports1) == len(reports2)
    equal_contents = set(report_hashes_1) == set(report_hashes_2)
    return equal_contents and equal_length

    
def generate_report(report_dict, article):
    report = Report(report_dict['Location'], report_dict['DateTime'], report_dict['EventTerm'],                    report_dict['SubjectTerm'], report_dict['Quantity'], article)
    return report

def compare_report_sets(expected_reports, generated_reports):
    expected_hashes = [r.report_hash() for r in expected_reports]
    generated_hashes = [r.report_hash() for r in generated_reports]
    print("==========Reports Not Generated==========")
    for h, r in zip(expected_hashes, expected_reports):
        if h not in generated_hashes:
            r.display()
    print("\n")
    print("==========Reports Erroneously Generated==========")
    for h, r in zip(generated_hashes, generated_reports):
        if h not in expected_hashes:
            r.display()
                
def run_tests(test_cases):
    cases_with_errors = []
    for t in test_cases:
        article = t['article']
        expected_reports = [generate_report(r, article) for r in t['reports']]
        generated_reports = process_article_new(article)
        if not compare_reports(expected_reports, generated_reports):
            cases_with_errors.append((article, expected_reports, generated_reports))
    error_proportion = len(cases_with_errors) / len(test_cases)
    print("==========Summary==========")
    print("% of cases with errors: {:.0f}%".format(error_proportion * 100))
    print("===========================")
    print("\n")
    for error_case in cases_with_errors:
        print("==========Article Contents==========")
        print(error_case[0])
        print("\n")
        compare_report_sets(error_case[1], error_case[2])
        print("\n")

test_cases = []

article = "Flash flooding across Afghanistan and Pakistan has left more than 160 dead and dozens stranded in one of South Asia's worst natural disasters this year, say officials.  The flooding, caused by unusually heavy rain, has left villagers stuck in remote areas without shelter, food or power.  Mountainous Afghanistan was the worst hit, with 61 people killed and approximately 500 traditional mud-brick homes washed away in more than a dozen villages in Sarobi, a rural district less than an hour from Kabul, officials said.  Floods left a village devastated in the remote eastern Afghan province of Nuristan. At least 60 homes were destroyed across three districts, said provincial spokesman Mohammad Yusufi. No one was killed.  Authorities have been unable to deliver aid to some badly affected villages by land as roads in the area are controlled by the Taliban, Yusufi added.  “We have asked the national government for help as have an overwhelming number of locals asking for assistance, but this is a Taliban-ridden area,” Yusufi said.  At least 24 people were also died in two other eastern border provinces, Khost and Nangarhar, according to local officials. More than fifty homes and shops were destroyed and thousands of acres of farmland flooded.  In Pakistan monsoon rains claimed more than 80 lives, local media reported. Houses collapsing, drowning and electrocution all pushed up the death toll, said Sindh Information Minister Sharjeel Inam Memon.  In Karachi, the commercial capital and a southern port city that is home to 18 million people, poor neighborhoods were submerged waist-deep in water and many precincts suffered long power outages. Deaths were also reported in the north and west of the country.  Additional reporting by Reuters"
expected_reports = []
expected_reports.append(Report(['Afghanistan', 'Pakistan'], ['this year'], 'die', 'person', 160, '').to_json())
expected_reports.append(Report(['Afghanistan', 'Pakistan'], ['this year'], 'strand', 'person', 'dozens', '').to_json())
expected_reports.append(Report(['Afghanistan', 'Pakistan'], ['this year'], 'stick', 'villager', None, '').to_json())
expected_reports.append(Report(['Sarobi'], ['this year'], 'kill', 'people', 61, '').to_json())
expected_reports.append(Report(['Sarobi'], ['this year'], 'wash', 'home', 500, '').to_json())
expected_reports.append(Report(['Nuristan'], ['this year'], 'destroy', 'home', 60, '').to_json())
expected_reports.append(Report(['Khost', 'Nangarhar'], ['this year'], 'die', 'people', 24, '').to_json())
expected_reports.append(Report(['Khost', 'Nangarhar'], ['this year'], 'destroy', 'homes and shops', 50, '').to_json())
expected_reports.append(Report(['Pakistan'], ['this year'], 'die', 'people', 80, '').to_json())
expected_reports.append(Report(['Pakistan'], ['this year'], 'collapse', 'house', None, '').to_json())
d = {}
d['article'] = article
d['reports'] = expected_reports
test_cases.append(d)

article = "'Afghanistan state news agency, Bakhtar News Agency (BNA) report that at least 7 people have been killed in flash floods in Faryab Province in the north of the country. Flash floods in Baghlan Province have killed 1 person and injured around 10 others.  Flash floods struck on 08 May 2015 in Faryab Province after a period of heavy rainfall. The districts of Garyzan, Pashtunkot and Belcheragh were worst affected. BNA report that at least 7 people were killed and over 1,500 homes damaged. The Faizabada-Takhar highway have been closed to traffic and wide areas of crops and orchards have suffered damaged.  Kuwaiti News Agency (KUNA) also report that flooding struck in the Baghlan-i-Markazi district of Baghlan province, where 1 person was killed and several injured early on Saturday 09 May 2015.  “There was heavy rain in Baghlan-e-Markazi district Friday evening and the people left their houses to safer areas. It was early Saturday when a flash flood hit the area and washed away more than 500 houses,” district Governor Gohar Khan Babri told reporters in provincial capital Pul-e-Khumri, 160 km north of Kabul.'"
expected_reports = []
expected_reports.append(Report(['Faryab Province'], ['08 May 2015'], 'kill', 'people', 7, '').to_json())
expected_reports.append(Report(['Baghlan Province'], ['08 May 2015'], 'kill', 'person', 1, '').to_json())
expected_reports.append(Report(['Garyzan', 'Pashtunkot', 'Belcheragh'], ['08 May 2015'], 'kill', 'people', 7, '').to_json())
expected_reports.append(Report(['Garyzan', 'Pashtunkot', 'Belcheragh'], ['08 May 2015'], 'damage', 'home', '1,500', '').to_json())
expected_reports.append(Report(['Baghlan'], ['Saturday 09 May 2015'], 'kill', 'person', 1, '').to_json())
expected_reports.append(Report(['Baghlan-i-Markazi', 'Baghlan'], ['early Saturday'], 'wash', 'house', 500, '').to_json())
expected_reports.append(Report(['Baghlan-e-Markazi'], ['Friday evening'], 'leave', 'people', None, '').to_json())
d = {}
d['article'] = article
d['reports'] = expected_reports
test_cases.append(d)

article = "'ALGIERS (AA) – Hundreds of homes have been destroyed in Algeria‘s southern city of Tamanrasset following several days of torrential rainfall, a local humanitarian aid official said Wednesday.  The city was pounded by rainfall from March 19 to March 24, according to Ghanom Sudani, a member of a government-appointed humanitarian aid committee.  He added that heavy rains had destroyed as many as 400 residences.  “Hundreds of families have had to leave their homes after they were inundated with water,” Sudani told The Anadolu Agency.  www.aa.com.tr/en  Last month neighbouring Tunisia experienced heavy rainfall and flooding in Jendouba City.'"
expected_reports = []
expected_reports.append(Report(['Tamanrasset'], ['March 19'], 'destroy', 'homes', 'hundreds', '').to_json())
expected_reports.append(Report(['Tamanrasset'], ['March 19'], 'destroy', 'residence', 400, '').to_json())
expected_reports.append(Report(['Tamanrasset'], ['March 19'], 'leave', 'families', 'hundreds', '').to_json())
d = {}
d['article'] = article
d['reports'] = expected_reports
test_cases.append(d)

article = 'Heavy rain on Monday 09 March 2015 flooded at least 3 municipalities of Luanda, the capital of Angola.  According to Angola news agency ANGOP, Luanda fire department have reported the flooding has forced at least 800 families from their homes. Later reports suggest that as many as 1,770 homes have been damaged. The municipalities of Viana, Cacuaco and Belas are said to be the worst affected.  Some streets have been completely blocked by the floods, making it difficult for the authorities to carry out full assessments of the damage. Provincial deputy governor for technical area, Agostinho da Silva, told ANGOP that the government are providing assistance to those in flood affected areas, and have set up pumps to help remove the flood water.'
expected_reports = []
expected_reports.append(Report(['Luanda'], ['Monday 09 March 2015'], 'force', 'family', 800, '').to_json())
expected_reports.append(Report(['Luanda'], ['Monday 09 March 2015'], 'damage', 'home', '1,770', '').to_json())
d = {}
d['article'] = article
d['reports'] = expected_reports
test_cases.append(d)

article = 'Flooding in Albania has killed at least three people. Torrential rain caused power cuts and water outages. Schools were closed in the west and south of the country.  A 60-year-old man and his 26-year old daughter were found dead after their car was swept away by floodwaters in Lac, northwest of the capital, Tirana, late Tuesday.  A 21-year-old motorcycle driver was also found dead in Lac, while his teenage passenger was rescued.  Army troops were on standby to help emergency workers with evacuation efforts.  “The children were screaming and crying,” said one unidentified woman whose house was flooded. “I did not know what to do. We decided to put them in a room in the second floor where it is higher.”  Authorities have evacuated families from five buildings.  The flooding hindered hospital and other public services and damaged a large area of farmland.  As the bad weather continued, the number of affected areas increased throughout Wednesday.  In neighboring Greece, weather warnings were issued for nearby parts of the country.'
expected_reports = []
expected_reports.append(Report(['Albania'], ['late Tuesday'], 'kill', 'people', 'three', '').to_json())
expected_reports.append(Report(['Tirana'], ['late Tuesday'], 'evacuate', 'building', 'five', '').to_json())
expected_reports.append(Report(['Tirana'], ['late Tuesday'], 'flood', 'house', None, '').to_json())
d = {}
d['article'] = article
d['reports'] = expected_reports
test_cases.append(d)

article = "Nineteen people are feared dead after violent storms and severe flooding swept the French Riviera, including three people who drowned in a retirement home after a river broke its banks."
expected_reports = []
expected_reports.append(Report(None, [], 'fear dead', 'people', 'nineteen', '').to_json())
expected_reports.append(Report(None, [], 'drown', 'people', 'three', '').to_json())
d = {}
d['article'] = article
d['reports'] = expected_reports
test_cases.append(d)

article = "More than fifty homes and shops were destroyed and thousands of acres of farmland flooded."
expected_reports = []
expected_reports.append(Report(None, [], 'destroy', 'homes and shops', 'fifty', '').to_json())
d = {}
d['article'] = article
d['reports'] = expected_reports
test_cases.append(d)

article = "Quoting an official from the Badakhshan provincial government, Xinhua also said that the foods had damaged or destroyed more than 120 houses in the district."
expected_reports = []
expected_reports.append(Report(['Badakhshan'], [], 'destroy', 'house', 120, '').to_json())
d = {}
d['article'] = article
d['reports'] = expected_reports
test_cases.append(d)

article = article = "The June 17 tornado whipped through Essa Township around the supper hour, leaving 100 families homeless while others had to clean up downed trees and debris."
expected_reports = []
expected_reports.append(Report(['Essa Township'], ['June 17'], 'leave homeless', 'family', 100, '').to_json())
d = {}
d['article'] = article
d['reports'] = expected_reports
test_cases.append(d)

article = "Mountainous Afghanistan was the worst hit, with 61 people killed and approximately 500 traditional mud-brick homes washed away in more than a dozen villages in Sarobi, a rural district less than an hour from Kabul, officials said."
expected_reports = []
expected_reports.append(Report(['Sarobi'], [], 'kill', 'people', 61, '').to_json())
expected_reports.append(Report(['Sarobi'], [], 'wash', 'home', 500, '').to_json())
d = {}
d['article'] = article
d['reports'] = expected_reports
test_cases.append(d)

article = "Further severe weather, floods and landslides have left 14 people dead and 4 missing in southern China.  Yesterday the Chinese government said that the storms and heavy rainfall from 18 to 22 June 2014 affected nine southern provinces. 8,700 homes have been destroyed, 66,000 homes damaged and forced 337,000 people to evacuate. 42,000 hectares of crops have also been destroyed. Further heavy rainfall is forecast for the next 24 hours."
expected_reports = []
expected_reports.append(Report(['China'], [], 'leave dead', 'people', 14, '').to_json())
expected_reports.append(Report(['China'], ['18 June 2014'], 'destroy', 'home', '8,700', '').to_json())
expected_reports.append(Report(['China'], ['18 June 2014'], 'damage', 'home', '66,000', '').to_json())
expected_reports.append(Report(['China'], ['18 June 2014'], 'force', 'people', '337,000', '').to_json())
d = {}
d['article'] = article
d['reports'] = expected_reports
test_cases.append(d)

article = "It was early Saturday when a flash flood hit the area and washed away more than 500 houses"
expected_reports = []
expected_reports.append(Report(None, ['early Saturday'], 'wash', 'house', 500, '').to_json())
d = {}
d['article'] = article
d['reports'] = expected_reports
test_cases.append(d)

article = "Within hours of the storm, Dowdall had declared a state of emergency and brought in Essa Township emergency departments staff, as well Simcoe County administrators, to assist the 300 people displaced by the storm."
expected_reports = []
expected_reports.append(Report(['Essa Township', 'Simcoe County'], [], 'displace', 'people', 300, '').to_json())
d = {}
d['article'] = article
d['reports'] = expected_reports
test_cases.append(d)

article = "BEIJING, March 31 (Xinhua) -- The Ministry of Civil Affairs has sent 1,000 tents, 2,000 sleeping bags, 2,000 folding beds and 1,000 sets of folding desks and chairs to Jianhe County in southwestern Guizhou Province after it was hit by a 5.5-magnitude earthquake on Monday morning.  No deaths have been reported, though the quake was Guizhou's biggest in terms of magnitude since 1949. More than 23,000 people have been affected and 2,536 relocated.  Provincial authorities have sent teams to help with the rescue work and allocated 1 million yuan (about 162,880 U.S. dollars) and 206 tents for disaster relief."
expected_reports = []
expected_reports.append(Report(['Guizhou Province'], ['Monday morning'], 'relocate', 'person', '2,536', '').to_json())
expected_reports.append(Report(['Guizhou Province'], ['Monday morning'], 'affect', 'people', '23,000', '').to_json())
d = {}
d['article'] = article
d['reports'] = expected_reports
test_cases.append(d)

article = 'As many as 2,214 households have been affected by the rainstorms in Rio Grande do Sul, the Emergency Management Service reported today (Dec. 28). A total of 1,964 households were displaced. The storms hit forty municipalities.  According to the government of Rio Grande do Sul, the State Coordination for Emergency Management continues to monitor and provide assistance to the impacted municipalities and communities.  Last Saturday (26), President Rousseff flew over the region, which borders Argentina and Uruguay, and announced the provision of $6.6 million to help communities hit by the floods.  This has been the fifth flood this year in the state, and the most severe. The Quaraí river rose a record 15.28 meters. The situation got even worse with the rise of the Uruguay river.  The rainstorm has disrupted rice harvest in the municipality of Quaraí and caused the Quaraí-Artigas international bridge between Brazil and Uruguay to remain closed off for 22 hours.    Translated by Mayra Borges'
expected_reports = []
expected_reports.append(Report(['Rio Grande do Sul'], ['Dec. 28'], 'displace', 'household', '1,964', '').to_json())
expected_reports.append(Report(['Rio Grande do Sul'], ['Dec. 28'], 'affect', 'household', '2,214', '').to_json())
d = {}
d['article'] = article
d['reports'] = expected_reports
test_cases.append(d)

article = "Verified  Kampong Cham, Kratie, Stung Treng and Kandal  Description  Due to high intensity of rainfall, Mekong River has swell and caused flooding to the surrounding areas. More flooding is expected if the rain continues. The provinces affected so far includes: Kampong Cham, Kratie, Stung Treng and Kandal12 out of Cambodia's 25 cities and provinces are suffering from floods caused by monsoon rains and Mekong River floodingIMPACT45 dead16,000 families were affected and evacuated3,080 houses inundated44,069 hectares of rice field were inundated5,617 hectares of secondary crops were inundatedRESPONSEThe local authorities provided response to the affected communities. More impact assessment is still conducted by provincial and national authorities.The government also prepared 200 units of heavy equipment in Phnom Penh and the provinces of Takeo, Svay Rieng, Oddar Meanchey and Battambang to divert water or mitigate overflows from inundated homes and farmland"
expected_reports = []
expected_reports.append(Report(['Cambodia'], [], 'inundate', 'house', '3,080', '').to_json())
expected_reports.append(Report(['Cambodia'], [], 'evacuate', 'family', '16,000', '').to_json())
d = {}
d['article'] = article
d['reports'] = expected_reports
test_cases.append(d)

article = "No one was killed."
expected_reports = []
d = {}
d['article'] = article
d['reports'] = expected_reports
test_cases.append(d)

reports = process_article_new("'ALGIERS (AA) – Hundreds of homes have been destroyed in Algeria‘s southern city of Tamanrasset following several days of torrential rainfall, a local humanitarian aid official said Wednesday.  The city was pounded by rainfall from March 19 to March 24, according to Ghanom Sudani, a member of a government-appointed humanitarian aid committee.  He added that heavy rains had destroyed as many as 400 residences.  “Hundreds of families have had to leave their homes after they were inundated with water,” Sudani told The Anadolu Agency.  www.aa.com.tr/en  Last month neighbouring Tunisia experienced heavy rainfall and flooding in Jendouba City.'")
for r in reports:
    r.display()

run_tests(test_cases)

