import re
import requests
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
from sklearn import metrics
import matplotlib.pyplot as plt
from IPython.display import IFrame
from sklearn.cluster import KMeans
import matplotlib.font_manager as fm
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle, Circle, Arc

#plt.style.use('ggplot')
get_ipython().magic('matplotlib inline')
pd.options.display.max_columns=25

import mechanize

import cookielib
import json

cap_url = "http://stats.nba.com/stats/commonallplayers?"
cap_param = {'IsOnlyCurrentSeason':"1",
                      'LeagueID': u'00',
                      'Season': u'2015-16'}
cap_resp = requests.get(cap_url, params=cap_param)
if cap_resp.status_code != 200:
    br = mechanize.Browser()
    cj = cookielib.LWPCookieJar()
    br.set_cookiejar(cj)
    br.set_handle_equiv(True)
    br.set_handle_gzip(True)
    br.set_handle_redirect(True)
    br.set_handle_referer(True)
    br.set_handle_robots(False)
    br.set_handle_refresh(mechanize._http.HTTPRefreshProcessor(), max_time=1)
    br.addheaders = [('User-agent', 'Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.0.1) Gecko/2008071615 Fedora/3.0.1-1.fc9 Firefox/3.0.1')]
    data = br.open(cap_resp.url).get_data()
    jdata = json.loads(data)
else:
    jdata = cap_resp.json()
cap_df = pd.DataFrame(jdata['resultSets'][0]['rowSet'],columns=jdata['resultSets'][0]['headers'])
cap_df.head()

def format_name_last_comma_first(name):
    if len(name.split())<2:
        return 'All'
    if ',' in name:
        names = name.split(',')
    else:
        names = name.split()
    return names[-1].strip()+', '+' '.join(names[0:-1]).strip()
print(format_name_last_comma_first('James, Lebron'))

def player_lookup(name='All',season='2015-16'):
    cap_url = "http://stats.nba.com/stats/commonallplayers?"
    cap_param = {'IsOnlyCurrentSeason':"1",
                      'LeagueID': u'00',
                      'Season': season}
    cap_resp = requests.get(cap_url, params=cap_param)
    cap_df = pd.DataFrame(cap_resp.json()['resultSets'][0]['rowSet'],columns=cap_resp.json()['resultSets'][0]['headers'])
    return cap_df

pname = 'Blake Griffin'
cap_df = player_lookup()
pinfo = cap_df[cap_df['DISPLAY_LAST_COMMA_FIRST']==format_name_last_comma_first(pname)]
pID = pinfo['PERSON_ID'].values
print(pinfo.head())
print(pID)

sc_params = {u'AheadBehind': u'',
 u'ClutchTime': u'',
 u'ContextFilter': u'',
 u'ContextMeasure': u'FGA',
 u'DateFrom': u'',
 u'DateTo': u'',
 u'EndPeriod': u'',
 u'EndRange': u'',
 u'GameID': u'',
 u'GameSegment': u'',
 u'LastNGames': 0,
 u'LeagueID': u'00',
 u'Location': u'',
 u'Month': 0,
 u'OpponentTeamID': 0,
 u'Outcome': u'',
 u'Period': 0,
 u'PlayerID': pID,
 u'PointDiff': u'',
 u'Position': u'',
 u'RangeType': u'',
 u'RookieYear': u'',
 u'Season': u'2015-16',
 u'SeasonSegment': u'',
 u'SeasonType': u'Regular Season',
 u'StartPeriod': u'',
 u'StartRange': u'',
 u'TeamID': 0,
 u'VsConference': u'',
 u'VsDivision': u''}

sc_url = 'http://stats.nba.com/stats/shotchartdetail?'
sc_resp = requests.get(sc_url,params=sc_params)
print(sc_resp.url) 
if sc_resp.status_code != 200:
    br = mechanize.Browser()
    cj = cookielib.LWPCookieJar()
    br.set_cookiejar(cj)
    br.set_handle_equiv(True)
    br.set_handle_gzip(True)
    br.set_handle_redirect(True)
    br.set_handle_referer(True)
    br.set_handle_robots(False)
    br.set_handle_refresh(mechanize._http.HTTPRefreshProcessor(), max_time=1)
    br.addheaders = [('User-agent', 'Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.0.1) Gecko/2008071615 Fedora/3.0.1-1.fc9 Firefox/3.0.1')]
    data = br.open(sc_resp.url).get_data()
    jdata = json.loads(data)
else:
    jdata = sc_resp.json()
player_shot_df = pd.DataFrame(jdata['resultSets'][0]['rowSet'],columns=jdata['resultSets'][0]['headers'])
print(player_shot_df.shape)
player_shot_df.head(5)

#Get all missed shots from mid-range, 16-24ft.
miss_mid_df = player_shot_df[(player_shot_df['EVENT_TYPE']=='Missed Shot') &
                             (player_shot_df['SHOT_ZONE_BASIC']=='Mid-Range')]
miss_mid_df.head(10)

miss_mid_df['SHOT_ZONE_AREA'].value_counts()

cnt_miss_mid_df=miss_mid_df[miss_mid_df['SHOT_ZONE_AREA']=='Center(C)']
cnt_miss_mid_df.head(6)

for index, row in cnt_miss_mid_df.iterrows():
    url = 'http://stats.nba.com/stats/locations_getmoments/?eventid='+ str(row['GAME_EVENT_ID']) +             '&gameid=' + str(row['GAME_ID'])
    print(url)
    response = requests.get(url)
    if response.status_code != 200:
        br = mechanize.Browser()
        cj = cookielib.LWPCookieJar()
        br.set_cookiejar(cj)
        br.set_handle_equiv(True)
        br.set_handle_gzip(True)
        br.set_handle_redirect(True)
        br.set_handle_referer(True)
        br.set_handle_robots(False)
        br.set_handle_refresh(mechanize._http.HTTPRefreshProcessor(), max_time=1)
        br.addheaders = [('User-agent', 'Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.0.1) Gecko/2008071615 Fedora/3.0.1-1.fc9 Firefox/3.0.1')]
        data = br.open(response.url).get_data()
        jdata = json.loads(data)
    else:
        jdata = response.json()
    moments = jdata["moments"]
    moment_data = []
    for m in moments:
        for p in m[5]:
            p+=[moments.index(m), m[2],m[3]]
            moment_data.append(p)
    event_track_dict
pm_df = pd.DataFrame(data=moment_data,columns=['teamID','playerID','xloc','yloc','zloc','moment','gametime','shottime'])
pm_df.head(15)

def get_moments(gameid=None,eventid=None):
    if gameid is None or eventid is None:
        ValueError('Must supply a gameid and a eventid')
    url = 'http://stats.nba.com/stats/locations_getmoments/?eventid='+ str(eventid) +             '&gameid=' + str(gameid)
    print(url)
    response = requests.get(url)
    moments = response.json()["moments"]
    moment_data = []
    for m in moments:
        for p in m[5]:
            p+=[moments.index(m), m[2],m[3]]
            moment_data.append(p)
    pm_df = pd.DataFrame(data=moment_data,columns=['teamID','playerID','xloc','yloc','zloc','moment','gametime','shottime'])
    return pm_df

