get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
# Change these to True to set up the DB the first time
i_know_this_will_delete_everything = True
initialize_id_test = True
initialize_id = False

import os
import sys
from sqlalchemy import create_engine
from sqlalchemy import exc
from sqlalchemy import func
from sqlalchemy import Table, text
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from internal_displacement.model.model import Status, Session, Category, Article, Content, Country, CountryTerm,     Location, Report, ReportDateSpan, ArticleCategory, Base 
    
def init_db(db_url, i_know_this_will_delete_everything=False):
    """
    Warning! This will delete everything in the database!
    :param session: SQLAlchemy session
    """
    if not i_know_this_will_delete_everything:
        raise RuntimeError("Tried to init_db without knowing it would delete everything!")
    sql_path = '../docker/localdb/schema.sql'
    with open(sql_path, 'r') as schema:
        session.execute(text(schema.read()))
    session.commit()
    
db_host = os.environ.get('DB_HOST')

if initialize_id:
    db_url = 'postgresql://{user}:{password}@{db_host}/{db}'.format(
        user='jupyter', password='jupyter', db_host=db_host, db='id')
    
if initialize_id_test:
    db_url = 'postgresql://{user}:{password}@{db_host}/{db}'.format(
        user='jupyter', password='tester', db_host=db_host, db='id_test')
    
engine = create_engine(db_url)
Session.configure(bind=engine)
session = Session()
    
init_db(db_url, i_know_this_will_delete_everything=i_know_this_will_delete_everything)

import spacy
import json
from datetime import datetime
from internal_displacement.scraper import Scraper
from internal_displacement.interpreter import Interpreter
from internal_displacement.pipeline import Pipeline
from internal_displacement.add_countries import load_countries, delete_countries
import pandas as pd

from internal_displacement.pipeline import get_coordinates_mapzen

# Pre-load list of countries into the database
load_countries(session)

scraper = Scraper()
nlp = spacy.load('en')
person_reporting_terms = [
    'displaced', 'evacuated', 'forced', 'flee', 'homeless', 'relief camp',
    'sheltered', 'relocated', 'stranded', 'stuck', 'stranded', "killed", "dead", "died", "drown"
]

structure_reporting_terms = [
    'destroyed', 'damaged', 'swept', 'collapsed',
    'flooded', 'washed', 'inundated', 'evacuate'
]

person_reporting_units = ["families", "person", "people", "individuals", "locals", "villagers", "residents",
                            "occupants", "citizens", "households", "life"]

structure_reporting_units = ["home", "house", "hut", "dwelling", "building", "shop", "business", "apartment",
                                     "flat", "residence"]

relevant_article_terms = ['Rainstorm', 'hurricane',
                          'tornado', 'rain', 'storm', 'earthquake']
relevant_article_lemmas = [t.lemma_ for t in nlp(
    " ".join(relevant_article_terms))]

data_path = '../data'

interpreter = Interpreter(nlp, person_reporting_terms, structure_reporting_terms, person_reporting_units,
                          structure_reporting_units, relevant_article_lemmas, data_path,
                          model_path='../internal_displacement/classifiers/default_model.pkl',
                          encoder_path='../internal_displacement/classifiers/default_encoder.pkl')

# Load the pipeline
pipeline = Pipeline(session, scraper, interpreter)

# Load the test urls
test_urls = pd.read_csv('../data/idmc_uniteideas_training_dataset.csv')
test_urls = test_urls['URL'].tolist()

# Process the first 40 urls
for url in test_urls[20:25]:
    try:
        pipeline.process_url(url)
    except exc.IntegrityError:
        session.rollback()

print("{} articles in database".format(session.query(Article.id).count()))

article_stats = session.query(Article.status, func.count(Article.status)).group_by(Article.status).all()
print("Article statuses:")
for status, ct in article_stats:
    print("{}: {}".format(status, ct))

url = 'http://floodlist.com/africa/torrential-rains-destroy-400-homes-in-algeria'

article = session.query(Article).filter_by(url=url).first()

print("Status: {}".format(article.status))
print("Domain: {}".format(article.domain))
print("Language: {}".format(article.language))
print("Relevance: {}".format(article.relevance))
print("Category: {}".format(article.categories[0].category))
print("Num Reports: {}".format(len(article.reports)))

report = article.reports[0]

print("Event: {}".format(report.event_term))
print("Subject: {}".format(report.subject_term))
print("Quantity: {}".format(report.quantity))

# Datespan
date_span = report.datespans[0]
print("Report covers period from {} to {}".format(date_span.start, date_span.finish))

# Location
locations = report.locations
print("{} locations found".format(len(locations)))

location = locations[0]
country = location.country
print("Location: {}".format(location.description))
print("City: {}".format(location.city))
print("State: {}".format(location.subdivision))
print("Country code: {}".format(country.code))
print("Country name(s): {}".format([t.term for t in country.terms]))

