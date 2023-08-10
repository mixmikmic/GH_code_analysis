import pandas as pd
import numpy as np
import datetime
get_ipython().magic('matplotlib inline')
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
plt.style.use('newsprint.mplstyle')

import sys
sys.path.append('../analysis')
import plotting
import text
from speeches import Speech, SpeechCollection

inaugural_speeches = SpeechCollection(['../data/tapp/inaugurals.json']).speeches

def iter_rows(speeches, stopwords=None):
    for speech in speeches:
        total_word_count = sum(1 for _ in text.tokenize(speech.text))
        content_word_count = sum(1 for _ in text.tokenize(speech.text, stopwords))
        yield dict(title=speech.title, 
                   author=speech.author, 
                   timestamp=speech.timestamp,
                   total_word_count=total_word_count, 
                   content_word_count=content_word_count)

df = pd.DataFrame(iter_rows(inaugural_speeches, text.standard_stopwords))
df['timestamp'] = pd.to_datetime(df.timestamp)
df['content_word_proportion'] = df.content_word_count / df.total_word_count

fig = plt.figure(figsize=(8.5, 5.5))
ax = fig.gca()
# matplotlib can't handle np.timestamp64 types
# pandas can't handle dual axes (at least, not well)
# so we meet in the awkward middle
timestamps = df.timestamp.tolist()
term_days = 365*4.0
bar_kwargs = dict(width=term_days * 0.85, snap=False)
ax.bar(timestamps, df.total_word_count, label='Total words', color='darkgray', **bar_kwargs)
ax.bar(timestamps, df.content_word_count, label='Content words', color='darkgreen', **bar_kwargs)
ax.set_ylabel('Number of words')

ax2 = ax.twinx()
ax2.plot(timestamps, df.content_word_proportion, label='Percentage of total words\nthat are content words', 
         marker='.', markeredgecolor='darkred', markerfacecolor='darkred',
         color='red', linewidth=1, linestyle='dotted', alpha=1)
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
ax2.set_ylim(0, 1.0)
# ax2.set_ylim(0.2, 0.8)
ax2.set_ylabel('Percentage of totals words')

ticklabels = list(plotting.iter_inaugural_titles(inaugural_speeches))
ax.set_xticks(timestamps, minor=False)
ax.set_xticklabels(ticklabels, minor=False, rotation=90, size=8)
ax.set_xlabel('', visible=False)
plt.title('Inaugural address words')
ax.legend(loc='upper center')
ax2.legend(loc='upper right')
term_timedelta = pd.to_timedelta(term_days, 'D')
plt.xlim(timestamps[0] - term_timedelta, timestamps[-1] + term_timedelta)
plt.tight_layout()
# plt.savefig('Historical-inaugural-address-timeseries.pdf')

