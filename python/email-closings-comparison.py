import re
import sys
# Used fork of bigbang that doesn't crash when a malformed date appears
# Cloned from https://github.com/bjgfromthe703/bigbang and added to path
sys.path.append('/Users/brendan/data-delving/bigbang-fork')
from bigbang.utils import *
from bigbang.archive import Archive
from bigbang.thread import Thread
import warnings
import pandas as pd
import pickle
from collections import Counter
warnings.filterwarnings('ignore')

urls = ['http://mail.scipy.org/pipermail/ipython-dev/',
        'http://mail.scipy.org/pipermail/ipython-user/',
        'http://mail.scipy.org/pipermail/scipy-dev/',
        'http://mail.scipy.org/pipermail/scipy-user/',
        'https://lists.centos.org/pipermail/centos/',
        'https://mail.scipy.org/pipermail/numpy-discussion/',
        'http://lists.openstack.org/pipermail/openstack/',
        'https://mail.python.org/pipermail/python-list/',
        'http://lists.ucla.edu/pipermail/religionlaw/',
        'https://pidgin.im/pipermail/support/',
        'https://www.winehq.org/pipermail/wine-users/',
        'https://lists.freebsd.org/pipermail/freebsd-questions/',
        'https://mta.openssl.org/pipermail/openssl-users/',
        'https://mail.haskell.org/pipermail/beginners/',
        'https://mail.haskell.org/pipermail/haskell-cafe/',
        'https://lists.wikimedia.org/pipermail/wikitech-l/',
        'https://lists.blender.org/pipermail/bf-committers/',
        'https://lists.fedoraproject.org/pipermail/devel/',
        'https://lists.dns-oarc.net/pipermail/dns-operations/',
        'http://mailman.nginx.org/pipermail/nginx-devel/',
        'http://mailman.nginx.org/pipermail/nginx/'
        ]

count = 0
replyCount = 0
threads = []  # We will use these threads throughout
try:
    for x in range(len(urls)):
        archive = Archive(urls[x], archive_dir='archives')
        threads += archive.get_threads()
    for thread in threads:
        count += 1
        if thread.get_num_people() > 1:
            replyCount += 1
except Exception as e:
    print 'Error!', e
finally:
    with open('threads.p', 'wb') as outfile:  # Save for later
        pickle.dump(threads, outfile)
    print 'Total threads: ' + str(count)
    print 'Total threads w/ replies: ' + str(replyCount)
    print 'Baseline response rate: ' + '{:.1%}'.format(replyCount * 1.0 / count)

closingRegex = re.compile(r'\n((?:\w+\s+){0,2}\w+)(!+|,|\.)\n', re.IGNORECASE)

def getClosing(message, regex):
    matches = re.findall(regex, message)
    if matches:
        return matches[-1][0].lower()

closingDict = {}
try:
    for x in range(len(urls)):
        archive = Archive(urls[x], archive_dir='archives')
        myThreads = archive.get_threads()
        print 'Archive ' + urls[x] + ' has ' + str(len(myThreads)) +             ' threads; ' + str(len(archive.get_activity())) + ' participants'
    for thread in threads:
        initialMsg = thread.get_content()[0]
        initialMsg = clean_message(initialMsg)
        closing = getClosing(initialMsg, closingRegex)
        if closing is not None:
            if closing in closingDict:
                closingDict[closing] += 1
            else:
                closingDict[closing] = 1
except Exception as e:
    print e
finally:  # if something crashes, we can see the partial results!
    print '-' * 20 + '\nThese were the most frequent (possible) closings:'
    c = Counter(closingDict)
    for k, v in c.most_common(50):
        print '%s: %i' % (k, v)

newRegex = re.compile(r'\n(thanks|regards|cheers|best regards|'
    r'thanks in advance|thank you|best|kind regards|tia|enjoy|many thanks|'
    r'sincerely|thanks a lot|hth|bye|best wishes|thanks again|hope this helps|'
    r'thx|good luck|appreciated|all the best|thanks and regards|later|take care|'
    r'have fun|please help|yours|ciao|hope that helps|warm regards|with regards)'
    r'(!+|,|\.)\n', re.IGNORECASE)

newClosingDict = {}
try:
    for thread in threads:
        msgGotReplies = thread.get_num_people() > 1
        initialMsg = thread.get_content()[0]
        initialMsg = clean_message(initialMsg)
        closing = getClosing(initialMsg, newRegex)
        if closing is not None:
            if closing in newClosingDict:
                newClosingDict[closing]['count'] += 1
            else:
                newClosingDict[closing] = {'count': 1, 'replyCount': 0}
            if msgGotReplies:
                newClosingDict[closing]['replyCount'] += 1
except Exception as e:
    print e
finally:
    counts = []
    replyCounts = []
    replyRates = []
    closings = []
    for k, v in newClosingDict.iteritems():
        counts.append(v['count'])
        replyCounts.append(v['replyCount'])
        closings.append(k)
        replyRates.append(1.0 * v['replyCount'] / v['count'])
    d = {'numTimes': pd.Series(counts, index=closings),
         'replies': pd.Series(replyCounts, index=closings),
         'replyRate': pd.Series(replyRates, index=closings)}
    df = pd.DataFrame(d)

# Let's not wait half an hour every time we run this notebook
df = pd.DataFrame(d)
df.index.names = ['closing']
df.to_csv('closing_reply_rates.csv')

pd.options.display.float_format = '{:.3f}'.format
df = pd.DataFrame.from_csv('closing_reply_rates.csv')
df = df.sort_values(by='replyRate', ascending=False)
df

# Filter out closings without much of a sample size
df2 = df[df.numTimes >= 1000].sort('replyRate', ascending=False)
df2

# Graph it!
from bokeh.io import output_notebook, push_notebook, show
from bokeh.charts import Bar
from bokeh.charts.attributes import CatAttr
from bokeh.models import NumeralTickFormatter
output_notebook()
label = CatAttr(df=df2, sort=False)
bar = Bar(df2, values='replyRate', label=label, legend=None,
          ylabel='Reply Rate (%)', xlabel='Email Closing')
bar.yaxis.formatter = NumeralTickFormatter(format='0%')
handle = show(bar, notebook_handle=True)

def isThankful(closing):
    if 'thank' in closing:
        return 'Variation of thank you'
    return 'Not a variation of thank you'


grouped = df.drop('replyRate', 1).groupby(isThankful)
grouped = grouped.sum()
grouped['replyRate'] = grouped['replies'] / grouped['numTimes']
grouped

label = CatAttr(df=grouped, sort=False)
bar = Bar(grouped, values='replyRate', label=label, legend=None,
          ylabel='Reply Rate (%)', xlabel='Email Closing')
bar.yaxis.formatter = NumeralTickFormatter(format='0%')
handle = show(bar, notebook_handle=True)

thankCounter = {'replyCount': 0, 'count': 0}
otherCounter = {'replyCount': 0, 'count': 0}
try:
    for thread in threads:
        initialMsg = thread.get_content()[0]
        initialMsg = clean_message(initialMsg)
        hasReply = thread.get_num_people() > 1
        closing = getClosing(initialMsg, closingRegex)
        if closing is not None and 'thank' in closing:
            thankCounter['count'] += 1
            if hasReply:
                thankCounter['replyCount'] += 1
        else:
            otherCounter['count'] += 1
            if hasReply:
                otherCounter['replyCount'] += 1
except Exception as e:
    print e
finally:  # if something crashes, we can see the partial results!
    print 'Emails with thankful closings\n' + '-' * 20
    print 'Email count: ' + str(thankCounter['count'])
    print 'Reply count: ' + str(thankCounter['replyCount'])
    print 'Response rate: ' + str(thankCounter['replyCount'] * 1.0 /
                                  thankCounter['count'])
    print '\n'
    print 'All other emails\n' + '-' * 20
    print 'Email count: ' + str(otherCounter['count'])
    print 'Reply count: ' + str(otherCounter['replyCount'])
    print 'Response rate: ' + str(otherCounter['replyCount'] * 1.0 /
                                  otherCounter['count'])

