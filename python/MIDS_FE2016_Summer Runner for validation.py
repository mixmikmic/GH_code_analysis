# For our internal toolbox imports
import os
import sys
import logging
import pendulum as pd
import time
path_to_here = os.path.abspath('.')
NLP_PATH = path_to_here[:path_to_here.index('slack-pack') + 10] + '/code/'
sys.path.append(NLP_PATH)


from nlp.text import extractor as xt
from nlp.models.message_classification import SimpleClassifier
from nlp.utils.model_output_management import OutputHelper
from nlp.models.similarity_calculation import MessageSimilarity
from nlp.models.summarization import TFIDF as Model
from nlp.grammar import tokenizer as nt
from nlp.viz.cloud import Wordcloud

logger = logging.getLogger('MIDS_FE2016S_log')
logger.setLevel(logging.DEBUG)
LOGFILE = 'log/MIDS_FE2016S_log'
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# create console handler, set level of logging and add formatter
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)

# create file handler, set level of logging and add formatter
fh = logging.handlers.TimedRotatingFileHandler(LOGFILE, when='M', interval=1)
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(ch)
logger.addHandler(fh)

casdb = xt.CassandraExtractor(cluster_ips=['54.175.189.47'],
                              session_keyspace='test_keyspace',
                              table_name='awaybot_messages')

# Set up a query that gets about 100 messages from #general
tz = pd.timezone('US/Eastern')
week14 = pd.now(tz).subtract(weeks=21).timetuple()
week15 = pd.now(tz).subtract(weeks=25).timetuple()
week14_ts = 1468889336.0
week15_ts = 1466470136.0

print week15_ts
print week14_ts

mids_fe2016s_filter = (
    "SELECT * FROM fe_s16_messages WHERE"
    " ts > '{}' AND ts < '{}' "
    "AND CHANNEL = 'general' ALLOW FILTERING".format(week15_ts, week14_ts))

casdb.add_query("mids_fe2016s_filter", mids_fe2016s_filter)

rows = casdb.get_messages(
    "mids_fe2016s_filter")

for row in rows:
    print (row.id, row.text, row.author, row.team, row.url, row.timestamp)
    break
    
rows = casdb.get_messages(
    "mids_fe2016s_filter")
c = 0
for row in rows:
    c += 1
print c

# Run the model on that query and save the output vizualizations locally
FONT_PATH = NLP_PATH + 'nlp/data/font/Ranga-Regular.ttf'
IMG_FOLDER = NLP_PATH + 'nlp/data/img/'
msg_sim = MessageSimilarity()
msg_stream = casdb.get_messages(
    "mids_fe2016s_filter")
classifier = SimpleClassifier(message_similarity=msg_sim)
classified_window = classifier.classify_stream(msg_stream, low_threshold=.4, high_threshold=.7, low_step=.05, high_step=.02, max_messages=10000, verbose=False)
image_loader = OutputHelper()

uni_model = Model(window=classified_window, cleaner=nt.SimpleCleaner(), n_grams=2)
viz_topics = 0
for t, topic in enumerate(classified_window):  # one(?) per topic
    if len(topic) >= 3:
        # Generate the viz out of the model
        try:
            viz = Wordcloud(model=uni_model, document_id=t, max_words=(10, 5), font=FONT_PATH, multi_plot=True)
        except:
            logger.warning("Failed to generate word cloud for",exc_info=True)
            continue
        viz_topics += 1
        logger.info('topic {} for {} duration {} hour(s) has length {}'.format(t, 'general', 0, len(topic)))
        viz_path = IMG_FOLDER + 'FE2016S_{}_{}_{}_{}.png'.format('general', 'testing', 0, viz_topics)
        viz.save_png(viz_path, title='Topic {}'.format(viz_topics))
        logger.info('saved {}'.format(viz_path))

# Task one pseduo code
rows = casdb.get_messages(
    "mids_fe2016s_filter")

sorted_convo = []
for row in rows:
    sorted_convo.append((row.id, row.text, row.author, row.team, row.url, row.timestamp))
sorted_convo = sorted_convo.sort(key=lambda x: x[5])

from operator import itemgetter
sorted_convo = sorted(sorted_convo, key=itemgetter(5))

