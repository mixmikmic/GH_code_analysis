import time; print ("Last updated "  + time.strftime("%x"))

get_ipython().magic('load_ext skip_kernel_extension')
# https://stackoverflow.com/a/43584169/819544

import copy
try:
    import cPickle as pickle
except:
    import pickle
from datetime import datetime
import json
import string
import time

from bs4 import BeautifulSoup
import colorlover as cl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
from plotly.utils import PlotlyJSONEncoder
import praw
import requests

import warnings

warnings.filterwarnings('ignore') # Supress pandas warning in cell [27]
    
get_ipython().magic('matplotlib inline')

# If this cell produces an "IOPub data rate exceeded" warning, restart the
# notebook server using the following command:
#     jupyter notebook --NotebookApp.iopub_data_rate_limit=1.0e10
init_notebook_mode()

# Utility for serializing plotly figure objects to disk so we can transfer completed charts
# into a less technical report for broad consumption

def plotlyfig2json(fig, fpath=None):
    """
    Serialize a plotly figure object to JSON so it can be persisted to disk.
    Figure's persisted as JSON can be rebuilt using the plotly JSON chart API:
    
    http://help.plot.ly/json-chart-schema/
    
    If `fpath` is provided, JSON is written to file.
    
    Modified from https://github.com/nteract/nteract/issues/1229
    """

    redata = json.loads(json.dumps(fig.data, cls=PlotlyJSONEncoder))
    relayout = json.loads(json.dumps(fig.layout, cls=PlotlyJSONEncoder))

    fig_json=json.dumps({'data': redata,'layout': relayout})
  
    if fpath:
        with open(fpath, 'w',  encoding='utf-8-sig') as f:
            f.write(fig_json)
    else:
        return fig_json

from collections import namedtuple
import copy
import json
import requests
import time

class PushshiftAPI(object):
    """Minimal API"""
    base_url = 'https://api.pushshift.io/reddit/{}/search/'
    def __init__(self, max_retries=20, max_sleep=3600, backoff=2, 
                 max_results_per_request=500):
        self.max_retries = max_retries
        self.max_sleep   = max_sleep
        self.backoff     = backoff
        self.max_results_per_request = max_results_per_request
    def _epoch_utc_to_est(self, epoch):
        return epoch - 5*60*60
    def _wrap_thing(self, thing, kind):
        """Mimic praw.Submission and praw.Comment API"""
        thing['created'] = self._epoch_utc_to_est(thing['created_utc'])
        ThingType = namedtuple(kind, thing.keys())
        thing = ThingType(**thing)
        return thing
    def _get(self, kind, payload):
        if 'limit' not in payload:
            payload['limit'] = self.max_results_per_request
        url = self.base_url.format(kind)
        i, success = 0, False
        while (not success) and (i<self.max_retries) :
            rest = min(i*self.backoff, self.max_sleep)
            time.sleep(rest)
            response = requests.get(url, params=payload)
            success = response.status_code == 200
            i+=1
        return json.loads(response.text)['data']
    def _query(self, kind, stop_condition=lambda **x: False, **kwargs):
        limit = kwargs.get('limit', None)
        payload = copy.deepcopy(kwargs)
        n = 0
        while True:
            if limit is not None:
                if limit > self.max_results_per_request:
                    payload['limit'] = self.max_results_per_request
                    limit -= self.max_results_per_request
                else:
                    payload['limit'] = limit
                    limit = 0
            results = self._get(kind, payload)
            if len(results) == 0:
                return
            for thing in results:
                n+=1
                if stop_condition(**thing):
                    return
                thing = self._wrap_thing(thing, kind)
                yield thing
            payload['before'] = thing.created_utc 
            if (limit is not None) & (limit == 0):
                return
    def search_submissions(self, **kwargs):
        return self._query(kind='submission', **kwargs)
    def search_comments(self, **kwargs):
        return self._query(kind='comment', **kwargs)

use_pushshift = True

if use_pushshift:
    api = PushshiftAPI()
    results = api.search_submissions(q='megathread', subreddit='politics')
else:
    UA = 'listing megathreads, /u/shaggorama'
    r = praw.Reddit('me', user_agent=UA)
    results = r.subreddit('politics').search('megathread', sort='new', syntax='cloudsearch', time_filter='all')

pol_bot = 'PoliticsModeratorBot'

reviewed = []
vals = {'title':[], 'timestamp':[], 'score':[], 'num_comments':[], 'url':[], 'is_megathread':[]}
for i, subm in enumerate(results):
    reviewed.append(subm.id)
    is_polbot_megathread     = subm.author == pol_bot
    is_discussion_megathread = (subm.author != pol_bot) and subm.title.startswith("Discussion Megathread: ")
    is_supreme_court_decisions = 'Supreme Court decisions week of' in subm.title
    
    if (is_polbot_megathread or is_discussion_megathread) and not is_supreme_court_decisions:
        vals['title'].append(subm.title)
        vals['timestamp'].append(subm.created)
        vals['score'].append(subm.score)
        vals['num_comments'].append(subm.num_comments)
        vals['url'].append('http://reddit.com'+ subm.permalink)
        vals['is_megathread'].append(True)

# Get any pol_bot submissions that weren't captured in the "Megathread" query

if use_pushshift:
    user_subm = api.search_submissions(author=pol_bot, subreddit='politics')
else:
    user = r.redditor(pol_bot)
    user_subm = user.submissions.new(limit=None)

for subm in user_subm:
    if subm.id not in reviewed:
        reviewed.append(subm.id)
        if not 'Announcement: ShareBlue' in subm.title:
            print (subm.title)
            vals['title'].append(subm.title)
            vals['timestamp'].append(subm.created)
            vals['score'].append(subm.score)
            vals['num_comments'].append(subm.num_comments)
            vals['url'].append('http://reddit.com'+ subm.permalink)
            vals['is_megathread'].append(True)

# No real way to do this with pushshift. 
if not use_pushshift:
    source_articles = [
        'http://www.politico.com/story/2017/02/mike-flynn-russia-ties-investigation-235272',
        'http://www.politico.com/story/2017/02/paul-manafort-blackmail-russia-trump-235275',
        'http://www.independent.co.uk/news/world/americas/us-politics/donald-trump-tower-no-wiretap-barack-obama-house-intelligence-committee-chairman-devin-nunes-justice-a7638026.html',
        'http://money.cnn.com/2017/02/24/media/cnn-blocked-white-house-gaggle/',
        'http://www.independent.co.uk/news/world/americas/house-intelligence-committee-devin-nunes-donald-trump-paul-manafort-campaign-chairman-russia-ukraine-a7648546.html',
        'http://www.newyorker.com/news/ryan-lizza/the-continuing-fallout-from-trump-and-nuness-fake-scandal',
        'http://foreignpolicy.com/2017/04/21/trump-weighs-in-on-french-election-after-paris-terror-attack-marine-le-pen-emmanuel-macron-france/',
        'http://www.cnn.com/2017/04/21/politics/russia-trump-campaign-advisers-infiltrate/index.html',
        'https://mobile.nytimes.com/2017/04/24/world/europe/macron-russian-hacking.html',
        'http://www.cnn.com/2017/04/25/politics/michael-flynn-house-oversight-committee/',
        'http://www.thedailybeast.com/trump-exempts-entire-senior-staff-from-white-house-ethics-rules'
    ]

    for url in source_articles:
        search_url = 'https://www.reddit.com/r/politics/search.json'
        args = {
            'q':'url:'+ url,
            'restrict_sr':'on',
            'sort':'top',
            't':'all',
            'limit':1 # We just need the single top scoring post
        }

        response = requests.get(search_url, params = args, headers={'User-Agent':UA})
        data = json.loads(response.text)
        try:
            subm_j = data['data']['children'][0]['data']
        except:
            print ("No submission for url", url)
            continue
        reviewed.append(subm_j['id'])
        vals['title'].append(subm_j['title'])
        vals['timestamp'].append(subm_j['created'])
        vals['score'].append(subm_j['score'])
        vals['num_comments'].append(subm_j['num_comments'])
        vals['url'].append('http://reddit.com'+ subm_j['permalink'])
        vals['is_megathread'].append(False)

        time.sleep(2) # make sure we don't anger reddit the API

df = pd.DataFrame(vals)

# one last thing to consider: should I go though the top submissions listing to 
# look for anything else I missed that I might want to add?

# No real way to do this with pushshift. 
if not use_pushshift:
    med_score = df.score.median()
    med_num_comments = df.num_comments.median()
    min_date = df.timestamp.min()

    def clean_permalink(url):
        target = '?ref=search_posts'
        n = len(target)
        if target in url:
            url = url[:-n]
        return url

    df['url2'] = df.url.apply(clean_permalink)

    # Hand pick ones that look like we don't need
    skip_ids = [
        '62a3kj', # ‘Cards Against Humanity’ Creator Just Pledged To Buy and Publish Congress’s Browser History
        '5p3s9c', # Man Boasts Of Sexual Assault, Later Inaugurated 45th President Of United States
        '5ubnzs', # Admit it: Trump is unfit to serve
        '5uj24p', # California bill would make Election Day a state holiday
        '6dr6f8', # White Terrorists Killed More Americans This Week Than Refugees Have in 40 Years
        '692k30', # Sean Spicer thinks it's 'somewhat sad' people are still talking about the election. President Trump has mentioned it every day for the last week.
        '6971i4', # Pregnancy to cost 425% more under Donald Trump's health plan compared to Obamacare
        '5qvsey', # President Trump hits majority disapproval in record time, Gallup finds
    ]

    # ... this is going to add a bunch. Fuck it. Let's see how bad it gets.

    #vals = {'title':[], 'timestamp':[], 'score':[], 'num_comments':[], 'url':[]}
    for subm in r.subreddit('Politics').top('year', limit=50):
        #if (subm.created > min_date) and 'http://reddit.com'+subm.permalink not in df.url2.values:
        #    print subm.score, subm.num_comments, subm.id, subm.title
        if (subm.created > min_date) and            subm.id not in reviewed and            subm.id not in skip_ids:

            print (subm.id, subm.title)
            reviewed.append(subm.id)

            vals['title'].append(subm.title)
            vals['timestamp'].append(subm.created)
            vals['score'].append(subm.score)
            vals['num_comments'].append(subm.num_comments)
            vals['url'].append('http://reddit.com'+ subm.permalink)
            vals['is_megathread'].append(False)

    df = pd.DataFrame(vals)

get_ipython().run_cell_magic('skip', 'True', '\nurl = "http://www.gallup.com/poll/203198/presidential-approval-ratings-donald-trump.aspx"\ntarget_fig_title = "Do you approve or disapprove of the way Donald Trump is handling his job as president?"\n\nresponse = requests.get(url)\nsoup = BeautifulSoup(response.text)\nfigs = soup.findAll(\'figure\')\n\nurl = "http://www.gallup.com/poll/203198/presidential-approval-ratings-donald-trump.aspx"\ntarget_fig_title = "Do you approve or disapprove of the way Donald Trump is handling his job as president?"\n\nresponse = requests.get(url)\nsoup = BeautifulSoup(response.text)\nfigs = soup.findAll(\'figure\')\n\nfor f in figs:\n    try:\n        if f.find(\'figcaption\').find(\'div\').text == target_fig_title:\n            break\n    except:\n        continue\n\ntable = f.find(\'table\')')

# Source page: 
#   http://news.gallup.com/poll/201617/gallup-daily-trump-job-approval.aspx

#url = "http://www.gallup.com/viz/v1/xml/6b9df319-97d6-4664-80e1-ea7fe4672aca/POLLFLEXCHARTVIZ/TRUMPJOBAPPR201617.aspx"
url_daily  = 'http://news.gallup.com/viz/v1/xml/ad26ce43-218c-4a42-82de-ce878fa6d119/POLLFLEXCHARTVIZ/TRUMPJOBAPPR201617.aspx'
url_weekly = 'http://news.gallup.com/viz/v1/xml/ad26ce43-218c-4a42-82de-ce878fa6d119/POLLFLEXCHARTVIZ/CN349.aspx'

def parse_gallup_date(date_str):
    m, d, y = [int(x) for x in date_str.split('/')]
    return datetime(y,m,d)

def get_gallup_data(url):

    response = requests.get(url)
    soup = BeautifulSoup(response.text)

    data = soup.find('data').find('set').find('rs')

    gallup_dict = {'date':[], 'approval':[], 'disapproval':[]}
    for observation in data.findAll('r'):
        # ed = end date, sd = start date
        sd_date_str = observation['sd'] 
        ed_date_str = observation['ed'] 
        sd_date = parse_gallup_date(sd_date_str)
        ed_date = parse_gallup_date(ed_date_str)

        # I think associating the three day average with the middle date makes the most sense. 
        # Full disclosure: the gallup chart reports the end date as the date of the poll. This 
        # probably makes sense from their perspective of wanting to report the most recent 
        # polling figures, but I think doing it my way causes the spikes in polling results to
        # line up with events better (i.e. thing happens, polls swing the following day).
        #date = sd_date + (ed_date - sd_date)/2
        date = ed_date
        
        # fix bad date
        if date.year == 2015:
            date = datetime(2017, date.month, date.day)

        gallup_dict['date'].append(date)

        polls = observation.findAll('p')
        gallup_dict['disapproval'].append(int(float(polls[0].text)))
        gallup_dict['approval'].append(int(float(polls[1].text)))
    
    return pd.DataFrame(gallup_dict)

gallup_daily  = get_gallup_data(url_daily)
gallup_weekly = get_gallup_data(url_weekly)

gallup =pd.concat([gallup_daily[gallup_daily['date']<='2017-12-31'],
                   gallup_weekly[gallup_weekly['date']>'2017-12-31']
                  ])

def clean_title(title):
    prefixes = ["Megathread: ", "Discussion Megathread: ", "Discussion: "]
    suffixes = [" - Megathread"]
    for prefix in prefixes:
        if title.startswith(prefix):
            n = len(prefix)
            title = title[n:]
    for suffix in suffixes:
        if title.endswith(suffix):
            n = len(suffix)
            title = title[:-n]
    else:
        return title

df['title'] = df['title'].apply(clean_title)

df['date'] = pd.to_datetime(1000000000*df.timestamp)
#df.drop('timestamp', axis=1, inplace=True)
#df['hyperlink'] = '<a href="' + df['url'] + '">' + df['title'] + '</a>' 
## I'd like to make it so users can click on an item and jump to the submission, but it doesn't
## really work great in plotly so we'll skip that

# I like the red/blue colors in the 'RdYlBu' diverging palette, but the yellow is too light,
# so I'll replace that with the purple from the 'Accent' qualitative palette.
col_scale = np.asarray(cl.scales['3']['div']['RdYlBu']) 
col_scale[1] = cl.scales['3']['qual']['Accent'][1]

sctr_comments = go.Scatter(
    x = df['date'],
    y = df['num_comments'],
    text = df['title'],
    name = '# Comments',
    hoverinfo='text',
    mode ='markers',
    marker = dict(color=col_scale[2])
)

sctr_score = go.Scatter(
    x = df['date'],
    y = df['score'],
    text = df['title'],
    name = 'Submission Score',
    hoverinfo='text',
    mode ='markers',
    marker = dict(color=col_scale[0])
)
    
legend_at_bottom = dict(
        orientation = "h", # again, this does nothing. Vertical stacking occupies more space, but it works.
        xanchor = "center",
        x = 0.5
    )
    
layout = go.Layout(
    hovermode='closest', 
    title='Reddit Activity in /r/Politics',
    legend = legend_at_bottom
    )
    
fig = go.Figure(data=[sctr_comments, sctr_score], layout=layout)
iplot(fig)

# Put num_comments on separate axis
sctr_comments['y'] = -df['num_comments'] 
sctr_comments['yaxis'] = 'y2'

# Anchor observations to x-axis
sctr_comments['error_y'] = dict(
                            type='data',
                            array = df['num_comments'],
                            arrayminus= 0 * df['num_comments'],
                            color=col_scale[2],
                            width=0,
                            opacity=1
                            )

sctr_score['error_y'] = dict(
                            type='data',
                            array = 0*df['score'],
                            arrayminus= df['score'],
                            color=col_scale[0],
                            width=0,
                            opacity=1
                            )

series = [sctr_score, sctr_comments]
    
# get fancy with the axes.
# Will need to "hack" y-axis label offsets. Mostly works, but adds a '.' to the label
y1_max = df['score'].max()
y2_max = df['num_comments'].max()
y1_max_rnd = 1e3*int(y1_max/1e3)
y2_max_rnd = 1e3*int(y2_max/1e3)

layout['showlegend'] = False

layout['yaxis'] = dict(
        title=".                        Score",
        tickvals=(0, y1_max_rnd/2, y1_max_rnd ),
        range=(-y1_max - 3e3, y1_max + 3e3),
        zeroline=True,
        titlefont=dict(color = col_scale[0]),
        )

layout['yaxis2']=dict(
        title = "# Comments                        .",
        tickvals=[0, -y2_max/2, -y2_max],
        ticktext=[0, str(int(y2_max/2e3)) + 'k', str(int(y2_max/1e3)) + 'k'],
        range=(-y2_max - 3e3, y2_max + 3e3),
        overlaying='y',
        side='right',
        zeroline=False,
        titlefont=dict(color = col_scale[2]),
    )
            
fig = go.Figure(data=series, layout=layout)
plotlyfig2json(fig, 'reddit_activity_chart.json')
iplot(fig)

df.plot.scatter(x='num_comments', y='score')

print ("Correlation: ", df[['num_comments', 'score']].corr().iloc[0,1])

df['g_score'] = df['score'] / df['score'].max()
df['g_num_comments'] = df['num_comments'] / df['num_comments'].max()
df['rel_activity'] = df[['g_score', 'g_num_comments']].max(axis=1)

# Break up the error bar "anchors" by overlaying dashed white lines. 
# Would look better if we overlaid the scatter markers w/o error lines, but this is fine to 
# get the idea across.
residuals_muted = []
for i in range(len(df)):                                           
    record = df.ix[i]
    col_m = 'white'
    lty = 'dot'
    val, yax = record['score'], 'y'
    if record['g_score'] > record['g_num_comments']:
        val, yax = -record['num_comments'], 'y2'
    line = go.Scatter(dict(
        x = (record['date'], record['date']),
        y= (0, val),
        text = record['title'],
        hoverinfo='x',
        mode='lines',
        line=dict(
            color=col_m,
            dash=lty
            ),
        showlegend = False,
            yaxis=yax
    ))
    residuals_muted.append(line)
            
series.extend([sctr_score, sctr_comments])
series.extend(residuals_muted)

layout['showlegend'] =False
            
fig = go.Figure(data=series, layout=layout)
iplot(fig)

base_str = '{}% of submissions have g_score > g_num_comments' 
perc = 100*df[df['g_score'] < df['g_num_comments']].shape[0] / df.shape[0]

print (base_str.format(perc))

import matplotlib.pyplot as plt
df.plot.scatter(x='g_num_comments', y='g_score')
plt.plot([0, 1], [0, 1], 'r--')
plt.show()

ix = 1*(df['g_score'] > df['g_num_comments'])
df['g_col'] = col_scale[::-2][ix]

ix = 1*(df['g_score'] > df['g_num_comments'])
df['g_col'] = col_scale[::-2][ix]

df_a = df[ix==1]
df_b = df[ix==0]

reddit_activity_a = go.Scatter(
        x = df_a['date'],
        y = 100 * df_a['rel_activity'],
        text = df_a['title'],
        name = 'rel score > rel # comments',
        mode ='markers',
        hoverinfo ='text',
        marker = dict(color = df_a['g_col'])
    )

reddit_activity_b = go.Scatter(
        x = df_b['date'],
        y = 100 * df_b['rel_activity'],
        text = df_b['title'],
        name = 'rel score < rel # comments',
        mode ='markers',
        hoverinfo ='text',
        marker = dict(color = df_b['g_col'])
    )

series = [reddit_activity_a, reddit_activity_b]

layout = go.Layout(
    title='Relative Reddit activity, colored by whether or not score/max(score) > n_comments/max(n_comments)',
    hovermode='closest', 
    yaxis=dict(ticksuffix='%'),
    legend = legend_at_bottom
    )
    
fig = go.Figure(data=series, layout=layout)
iplot(fig)

ix = 1*(df['is_megathread'])
df['g_col'] = col_scale[::-2][ix]

df_a = df[ix==1]
df_b = df[ix==0]

reddit_activity_a = go.Scatter(
    x = df_a['date'],
    y = 100 * df_a['rel_activity'],
    text = df_a['title'],
    name = 'megathreads',
    mode ='markers',
    hoverinfo ='text',
    marker = dict(
        color= df_a['g_col'],
        line = dict(
            width = .5,
            color = 'rgb(0, 0, 0)'
            )
        ),
    error_y = dict(
        type='data',
        array = 0 * df_a['rel_activity'],
        arrayminus= 100 * df_a['rel_activity'],
        color=df_a['g_col'].iloc[0],
        width=0,
        opacity=1
        )
    )

series = [reddit_activity_a]

if len(df_b):
    reddit_activity_b = go.Scatter(
        x = df_b['date'],
        y = 100 * df_b['rel_activity'],
        text = df_b['title'],
        name = 'non-megathreads',
        mode ='markers',
        hoverinfo ='text',
        marker = dict(color = df_b['g_col']),
        error_y = dict(
            type='data',
            array = 0 * df_b['rel_activity'],
            arrayminus= 100 * df_b['rel_activity'],
            color=df_b['g_col'].iloc[0],
            width=0,
            opacity=1
            )
    )
    
    series = [reddit_activity_b, reddit_activity_a]


layout.title = 'Distribution of megathreads in our dataset'

fig = go.Figure(data=series, layout=layout)
iplot(fig)

# This is a little inelegant, but it gets the job done
terms = [
    'yates',
    'comey',
    'comeys',
    'flynn',
    'flynns',
    'russa',
    'russia',
    'russian',
    'russians',
    'investigation',
    'investigations',
    'trump-russia',
    'mueller',
    'manafort',
    'mccabe',
    'memo',
    'gates',
    'papadopoulos'
]

def strip_punct(s):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in s if ch not in exclude)

def is_russia_story(title):
    title_lwr = title.lower()
    tokens = [strip_punct(t) for t in title_lwr.split()]
    if any(t in terms for t in tokens):
        return True
    ## Not all Jeff Sessions stories or SIC stories are about Russia:
    ##   * 2017-02-10: Jeff Sessions confirmed as AG 
    ##   * 2017-05-11: SIC hearing on global threats (i.e. DPRK)
    ## Add Sessions hearing as a special case
    if 'sessions' in tokens and 'senate intelligence committee' in title_lwr:
        return True
    if 'special counsel' in title_lwr:
        return True
    return False

df['is_russia_story'] = df.title.apply(is_russia_story)

df['g_col'] = col_scale[::-2][ 1*df['is_russia_story'] ] 

# Encircle megathreads
df['line_color'] = df['g_col'].copy()
df['line_color'].loc[df['is_megathread']] =  'rgb(0, 0, 0)'

df_is_russia  = df.loc[df['is_russia_story']]
df_not_russia = df.loc[~df['is_russia_story']]

reddit_activity1 = go.Scatter(
    x = df_is_russia['date'],
    y = 100 * df_is_russia['rel_activity'],
    text = df_is_russia['title'],
    name = 'Russia Scandal',
    mode ='markers',
    hoverinfo ='text',
    marker = dict(
        color= df_is_russia['g_col'],
        line = dict(
            width = .5,
            color = df_is_russia['line_color']
            )
        ),
    error_y = dict(
        type = 'data',
        array      =   0 * df_is_russia['rel_activity'],
        arrayminus = 100 * df_is_russia['rel_activity'],
        color   = df_is_russia['g_col'].iloc[0],
        width   = 0,
        opacity = 1
        )
    )

reddit_activity2 = go.Scatter(
        x = df_not_russia['date'],
        y = 100 * df_not_russia['rel_activity'],
        name = 'Misc News',
        text = df_not_russia['title'],
        mode ='markers',
        hoverinfo ='text',
        marker = dict(
            color= df_not_russia['g_col'],
            line = dict(
                width = .5,
                color = df_not_russia['line_color']
                )
            ),
    error_y = dict(
        type = 'data',
        array      =   0 * df_not_russia['rel_activity'],
        arrayminus = 100 * df_not_russia['rel_activity'],
        color   = df_not_russia['g_col'].iloc[0],
        width   = 0,
        opacity = 1
        )
    )
    
# is_russia on top
series = [reddit_activity2, reddit_activity1]

layout['title'] = 'Relative Reddit Activity, highlight Russia scandal news and (megathreads encircled)'
layout['yaxis']['title'] = 'Relative Reddit Activity'
layout['yaxis']['ticksuffix']='%',

fig = go.Figure(data=series, layout=layout)

iplot(fig)

approval_sctr = go.Scatter(
    x = gallup['date'],
    y = gallup['approval'],
    text = gallup['approval'].apply(lambda x: str(x) + '%'),
    mode='lines',
    name = '% Approval',
    hoverinfo='text',
    line = dict(
        width = 2)
    )

disapproval_sctr = go.Scatter(
    x = gallup['date'],
    y = gallup['disapproval'],
    text = gallup['disapproval'].apply(lambda x: str(x) + '%'),
    mode='lines',
    name = '% Disapproval',
    hoverinfo='text',
    line = dict(
        width = 2)
    )

layout = go.Layout(
                   hovermode='closest', 
                   #legend=dict(orientation='h'),
                   title='Gallup polling for Trump (dis)approval',
                   yaxis = dict(
                        title='',
                        showgrid=False,
                        ticks='outside',
                        ticksuffix='%',
                        range=[30,70]
                       ),
                   xaxis = dict(
                        showgrid=False,
                        ticks='inside'
                        ),
                   shapes = [dict(
                        type='line',
                        x0=gallup['date'].iloc[0],
                        x1=gallup['date'].iloc[-1],
                        y0=50,
                        y1=50,
                        line=dict(
                            dash='dash',
                            width=.5
                            )
                        )]
                   )

series = [approval_sctr, disapproval_sctr]
fig = go.Figure(data=series, layout=layout)

iplot(fig)

gallup['perc_point_gap'] = gallup['disapproval'] - gallup['approval']
gallup['perc_diff_rel_aprvl'] = ( 100 * gallup['perc_point_gap'] / gallup['approval'] ).astype(np.int)

# Add fancy labels to our combined metric to communicate the two respective values it's based on
base_str = '{diff}% : {dis}% disapprove vs. {aprv}% approve'

build_str = lambda x: base_str.format(
    diff = x['perc_diff_rel_aprvl'],
    dis  = int(x['disapproval']),
    aprv = int(x['approval'])
)

gallup['rel_aprvl_str'] = gallup.apply(build_str, axis=1)

apprv_disapprv_diff_sctr = go.Scatter(
    x = gallup['date'],
    y = gallup['perc_diff_rel_aprvl'],
    text = gallup['rel_aprvl_str'],
    mode='lines',
    name = '% More Disapprove',
    hoverinfo='text',
    line = dict(
        color = col_scale[1], #'black',
        )
    )

most_recent_rel_diapproval = gallup['perc_diff_rel_aprvl'].iloc[-1]
base_string = "Latest polling indicates {}% more people disapprove of Trump than approve"
disapproval_string = base_string.format(most_recent_rel_diapproval)
    
layout['xaxis']['title'] = disapproval_string

max_rel_dis = np.ceil(gallup['perc_diff_rel_aprvl'].max()/10 +1)*10
layout['yaxis']['range'] = [-3, max_rel_dis]

series = [approval_sctr, disapproval_sctr, apprv_disapprv_diff_sctr]
fig = go.Figure(data=series, layout=layout)
plotlyfig2json(fig, 'gallup_chart.json')

iplot(fig)

del layout['shapes']
del layout['yaxis']['range']

apprv_disapprv_diff_sctr['yaxis'] = 'y2'
apprv_disapprv_diff_sctr['line'] = dict(color = col_scale[1], width=2)

layout['title'] = 'Response to U.S. Political Turmoil: Gallup Polling vs. /r/Politics Activity<br>(megathreads encircled)'

layout['legend'] = legend_at_bottom
layout['legend']['y'] = -.2 # Make more space to accomodate the subtitle

layout['yaxis']['title'] = 'Relative Reddit Activity'
layout['yaxis2'] = dict(
                    title='% More Disapprove',
                    ticksuffix='%',
                    titlefont=dict(color = col_scale[1]), # Color second y-axis title to match its associated series
                    overlaying='y',
                    side='right',
                )

# The goal here is to get the zeroline of both y-axes to line up.
# Having a lot of trouble scaling this automatically. Will probably cause me issues every time a range changes.
miny2 = gallup['perc_diff_rel_aprvl'].min()
maxy2 = gallup['perc_diff_rel_aprvl'].max()
miny1, maxy1 = (0,100)
layout.yaxis.range = (miny2, maxy1+3) # Probably going to cause issues in the future.
layout.yaxis2.range = (miny2, maxy2)
layout.yaxis.showgrid = False
layout.yaxis2.showgrid = True

series = [apprv_disapprv_diff_sctr, reddit_activity2, reddit_activity1]

fig = go.Figure(data=series, layout=layout)
plotlyfig2json(fig, 'main_chart.json')

iplot(fig)

