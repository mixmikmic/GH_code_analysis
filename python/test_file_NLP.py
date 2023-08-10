get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import pandas as pd
import spacy
from sklearn.metrics import classification_report
from internal_displacement.interpreter import Interpreter
from internal_displacement.excerpt_helper import Helper
from internal_displacement.excerpt_helper import MeanEmbeddingVectorizer
import gensim

### Set up necessary arguments for the Interpreter
nlp = spacy.load('en')
person_reporting_terms = [
    'displaced', 'evacuated', 'forced', 'flee', 'homeless', 'relief camp',
    'sheltered', 'relocated', 'stranded', 'stuck', 'accommodated']

structure_reporting_terms = [
    'destroyed', 'damaged', 'swept', 'collapsed',
    'flooded', 'washed', 'inundated', 'evacuate'
]

person_reporting_units = ["families", "person", "people", "individuals", "locals", "villagers", "residents",
                            "occupants", "citizens", "households"]

structure_reporting_units = ["home", "house", "hut", "dwelling", "building"]

relevant_article_terms = ['Rainstorm', 'hurricane',
                          'tornado', 'rain', 'storm', 'earthquake']
relevant_article_lemmas = [t.lemma_ for t in nlp(
    " ".join(relevant_article_terms))]

data_path = '../data'

# Initialize the interpreter
interpreter = Interpreter(nlp, person_reporting_terms, structure_reporting_terms, person_reporting_units,
                          structure_reporting_units, relevant_article_lemmas, data_path,
                          model_path='../internal_displacement/classifiers/default_model.pkl',
                          encoder_path='../internal_displacement/classifiers/default_encoder.pkl')

# Initializer the helper
helper = Helper(nlp, '../internal_displacement/classifiers/unit_vectorizer.pkl', 
               '../internal_displacement/classifiers/unit_model.pkl',
               '../internal_displacement/classifiers/term_vectorizer.pkl',
               '../internal_displacement/classifiers/term_model.pkl',
               '../internal_displacement/classifiers/terem_svc.pkl')

# Load the pre-trained Word2Vec model
w2v = gensim.models.KeyedVectors.load_word2vec_format('../data/GoogleNews-vectors-negative300.bin', binary=True)

#### Load the data and clean up the excerpts (removes text in brackets, irrelevant tokens etc.)

test_df = pd.read_excel("../data/IDETECT_test_dataset - NLP.csv.xlsx")
test_df['cleaned_text'] = test_df['excerpt'].apply(lambda x: helper.cleanup(x))

#### Extract reports and choose the most likely one
test_df['reports'] = test_df['cleaned_text'].apply(lambda x: interpreter.process_article_new(x))
test_df['most_likely_report'] = test_df['reports'].apply(lambda x: helper.get_report(x))

# Step 1. Use the rules-based Interpreter
test_df['unit_rules'] = test_df['most_likely_report'].apply(lambda x: x[2])

# Step 2. Use the classifier
X_test = helper.reporting_unit_vectorizer.transform(test_df['cleaned_text'])
test_df['unit_clf'] = helper.reporting_unit_classifier.predict(X_test)

# Step 3. Combine the predictions
test_df['unit_combined'] = test_df[['unit_rules', 'unit_clf']].apply(lambda x: helper.combine_predictions(x['unit_clf'], x['unit_rules']), axis=1)

# Step 1. Use the rules-based Interpreter
test_df['term_rules'] = test_df['most_likely_report'].apply(lambda x: x[1])

# Step 2. Use the classifiers to get the probabilities and combine them into a single prediction
w2vVectorizer = MeanEmbeddingVectorizer(w2v)
X_feat_1 = helper.reporting_term_vectorizer.transform(test_df['cleaned_text'])
p1 = helper.reporting_term_classifier.predict_proba(X_feat_1)
X_feat_2 = w2vVectorizer.transform(test_df['cleaned_text'])
p2 = helper.reporting_term_svc.predict_proba(X_feat_2)

test_df['term_clf'] = helper.combine_probabilities(p1, p2, helper.reporting_term_classifier.classes_)

# Step 3. Combine the predictions
test_df['term_combined'] = test_df[['term_rules', 'term_clf']].apply(lambda x: helper.combine_predictions(x['term_clf'], x['term_rules']), axis=1)

person_units = ["person", "people", "individuals", "locals", "villagers", "residents",
                "occupants", "citizens", "IDP"]

household_units = ["home", "house", "hut", "dwelling", "building", "families", "households"]

person_lemmas =[t.lemma_ for t in nlp(" ".join(person_units))]
household_lemmas =[t.lemma_ for t in nlp(" ".join(household_units))]

# Step 1. Get quantity from top report
test_df['quantity_rules_1'] = test_df['most_likely_report'].apply(lambda x: x[0])
test_df['quantity_rules_1'] = test_df['quantity_rules_1'].fillna(0)
test_df['quantity_rules_1'] = test_df['quantity_rules_1'].astype(int)

# Step 2. Get quantity using other rules
test_df['quantity_rules_2'] = test_df[['excerpt', 'unit_combined']].apply(lambda x: helper.get_number(x['excerpt'], x['unit_combined'], person_lemmas, household_lemmas), axis=1)
test_df['quantity_rules_2'] = test_df['quantity_rules_2'].fillna(0)
test_df['quantity_rules_2'] = test_df['quantity_rules_2'].astype(int)

test_df['quantity_combined'] = test_df[['quantity_rules_1', 'quantity_rules_2']].apply(lambda x: helper.combine_quantities(x['quantity_rules_1'], x['quantity_rules_2']), axis=1)

test_df['locations'] = test_df['excerpt'].apply(lambda x: interpreter.extract_countries(interpreter.cleanup(x)))

test_df['top_location'] = test_df['locations'].apply(lambda x: helper.choose_country(x))

test_df['loc_name'] = test_df['top_location'].apply(lambda x: x[0])
test_df['country_code'] = test_df['top_location'].apply(lambda x: x[1])

output_df = test_df[['excerpt_id', 'excerpt', 'unit_combined', 'term_combined', 
       'quantity_combined', 'loc_name', 'country_code']]

output_df.columns = ['excerpt_id', 'excerpt', 'reporting_unit', 'reporting_term', 
       'quantity', 'location_name', 'country_code']

output_df.to_csv('../data/test_NLP_output.csv', index=None)

