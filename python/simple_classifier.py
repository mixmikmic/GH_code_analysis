get_ipython().run_cell_magic('javascript', '', "document.getElementById('notebook-container').style.width = '95%'")

# Add the path to the slack-pack/code/ folder in order to be able to import nlp module
import sys, os

NLP_PATH = '/'.join(os.path.abspath('.').split('/')[:-1]) + '/'
sys.path.append(NLP_PATH)

get_ipython().magic('matplotlib inline')
from nlp.text import extractor as xt

from nlp.geometry import repr as gr
from nlp.geometry import dist as gd
from nlp.grammar import tokenizer as gt
from nlp.text import window as gw

from nlp.models import similarity_calculation as gsc
from nlp.models import message_classification as gmc

gr.list_corpora()

get_ipython().run_cell_magic('time', '', "# Initialize the GloVe representation\nglove100_rep = gr.GloVe('glove.6B.100d.txt')")

get_ipython().run_cell_magic('time', '', "# Initialize the GloVe representation\nglove300_rep = gr.GloVe('glove.6B.300d.txt')")

clean = gt.SimpleCleaner()

def dist_m2m(m1, m2):
    # tokenize
    text1 = clean(m1.lower())
    text2 = clean(m2.lower())

    # get geometric representation
    rep1 = glove100_rep(text1)
    rep2 = glove100_rep(text2)
    
    return gd.cosine(rep1, rep2)

def dist_m2m_300(m1, m2):
    # tokenize
    text1 = clean(m1.lower())
    text2 = clean(m2.lower())

    # get geometric representation
    rep1 = glove300_rep(text1)
    rep2 = glove300_rep(text2)
    
    return gd.cosine(rep1, rep2)

def inspect_window(window):
    print( 'Window has #{} topics\n'.format( len(window) ) )
    
    print( 'Topic length report:' )
    for i, tpc in enumerate(window):
        print( '  Topic #{:>2}  --> size: {:<3}'.format(i, len(tpc)) )

def print_topic(topic):
    for i,(m,r) in enumerate(topic):
        print '{} -- {}\n\t\033[33m{}\033[0m\n\n'.format(i,r,m.text)

def classify_stream(message_stream, distance=dist_m2m, max_messages=20,
                    low_threshold=.4, high_threshold=.7, low_step=.05, high_step=.02, verbose=True):
    topics = []
    for m, msg in enumerate(message_stream):
        if m > max_messages:
            m -= 1
            break

        if verbose:
            print '#{:>3}\033[33m ==> {}\033[0m'.format(m, msg.text.encode('ascii', 'ignore'))

        if len(topics) == 0:
            topics.insert(0, [(msg, 'First message')] )
            if verbose:
                print '\t First message (new 0)\n'

        else:
            # We will sequentially try to append to each topic ...
            #    as time goes by it is harder to append to a topic

            low_th = low_threshold
            high_th = high_threshold
            topic_scores = []  # in case no topic is close

            for t in xrange(len(topics)):
                tp_len = len(topics[t])
                distances = map(lambda x: distance(msg.text, x[0].text), topics[t])

                # Assign a non-linear score (very close messages score higher)
                score = sum([ 0 if d < low_th else 1 if d < high_th else 3 for d in distances ])

                # Very large topics (> 10) should be harder to append to,
                #   since the odds of a casual match are higher
                if (tp_len < 3):
                    if (score > 0):
                        reason = 'len({}) < 3 and distances({})'.format(tp_len, distances)
                        _topic = topics.pop(t)  # pop from topic queue
                        _topic.append( (msg, reason) )
                        topics.insert(0, _topic)  # append to first topic
                        if verbose:
                            print '\t inserted to #{} : {}\n'.format(t, reason)
                        break

                elif (tp_len < 10):
                    if (score > (tp_len - (2 - tp_len/15.) )):
                        reason = 'len({}) < 10 and distances({})'.format(tp_len, distances)
                        _topic = topics.pop(t)  # pop from topic queue
                        _topic.append( (msg, 'len({}) < 10 and distances({})'.format(tp_len, distances)) )
                        topics.insert(0, _topic)  # append to first topic
                        if verbose:
                            print '\t inserted to #{} : {}\n'.format(t, reason)
                        break

                elif (tp_len > 10):
                    if (score > tp_len*1.5):
                        reason = 'len({}) > 10 and distances({})'.format(tp_len, distances)
                        _topic = topics.pop(t)  # pop from topic queue
                        _topic.append( (msg, 'len({}) > 10 and distances({})'.format(tp_len, distances)) )
                        topics.insert(0, _topic)  # append to first topic
                        if verbose:
                            print '\t inserted to #{} : {}\n'.format(t, reason)
                        break

                topic_scores.append( (tp_len,score) )  # append score to topic_scores

                # else try with next topic --> harder
                low_th += low_step if low_th+low_step < high_th else high_step
                high_th += high_step
            else:
                # If no topic was suitable --> Start new topic
                topics.insert(0, [(msg, 'No similar topics (to 0) scores:({})'.format(topic_scores))] )
                if verbose:
                    print '\t No similar topics (new 0) scores:({})\n'.format(topic_scores)

    print '... Done, processed {} messages'.format(m)
    return topics

# Initialize the extractor (JSON or Cassandra)
awwdb = xt.CassandraExtractor(cluster_ips=['54.175.189.47'],
                              session_keyspace='test_keyspace',
                              table_name='awaybot_messages')

awwdb.list_channels()

# Need to call .get_messages each time, because if not the message_stream will have "dried out"
msg_stream = awwdb.get_messages(type_of_query='day', periods=5, channel='tech-stuff', min_words=5)

window_us = classify_stream(msg_stream, distance=dist_m2m, low_threshold=.4, high_threshold=.7, low_step=.05, high_step=.02, max_messages=30)

inspect_window(window_us)

print_topic(window_us[3])



from nlp.text import topic as gt
from nlp.text import window as gw

inspect_window(window_us)

real_window = gw.from_topic_list(window_us)

real_window.report_topics()

real_window.topics[1].report_messages()



import cPickle as pk

with open('../nlp/data/windows/alex_new_config_window.pk', 'wb') as f:
    pk.dump(real_window, f)



