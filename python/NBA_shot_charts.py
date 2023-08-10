import re
import requests
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import IFrame
import matplotlib.font_manager as fm

plt.style.use('ggplot')
get_ipython().magic('matplotlib inline')
pd.options.display.max_columns=25

url = 'http://stats.nba.com/stats/shotchartdetail?'        'CFID=&CFPARAMS=&ContextFilter=&ContextMeasure=FGA'        '&Counter=1000&DateFrom=&DateTo=&Direction=DESC'        '&GameID=0021500051&GameSegment=&LastNGames=0&LeagueID=00'        '&Location=&Month=0&OpponentTeamID=0&Outcome=&Period=0'        '&PlayerID=0&PlayerOrTeam=T&Position=&RookieYear='        '&Season=2015-16&SeasonSegment=&SeasonType=Regular+Season'        '&Sorter=PTS&TeamID=1610612744&VsConference=&VsDivision='
response = requests.get(url)
response.json().keys()

resource = response.json()["resource"]
resultSets = response.json()["resultSets"]
parameters = response.json()["parameters"]
print resource
parameters

print type(resultSets)

print len(resultSets)

print [type(r) for r in resultSets]

print [r.keys() for r in resultSets]

[r['name'] for r in resultSets]

resultSets[0]['headers']

resultSets[0]['rowSet'][0]

sc_df = pd.DataFrame(resultSets[0]['rowSet'],columns=resultSets[0]['headers'])
sc_df.head(8)

plt.style.use('seaborn-white')
fig = plt.figure(figsize=(8, 6))
scat = plt.scatter(x=sc_df.LOC_X,y=sc_df.LOC_Y,c=sc_df.SHOT_MADE_FLAG,
    cmap=plt.cm.RdYlGn, s=100, alpha=.6)
plt.show()

from IPython.display import Image
Image('img/example_nba_shotchart.png')

plt.style.use('seaborn-white')

fig,ax = plt.subplots(figsize=(8, 8))
scat = plt.scatter(x=sc_df.LOC_X,y=sc_df.LOC_Y,c=sc_df.SHOT_MADE_FLAG,
    cmap=plt.cm.RdYlGn, s=100, alpha=.6, zorder=1)
ax.invert_yaxis()

plt.style.use('grayscale')
#Read in the half court image
court = plt.imread("img/halfcourt.png")
#The baseline is 4 ft. from front of backboard. front of backboard is 15 in. from center of hoop (the origin).
#Therefore, my image extent needs to go from (4ft.+15in.) = 5.25 ft. so 52.5 ft. in the coordinate system.
img = plt.imshow(court, zorder=0, extent=[-250,250,420,-52.5])

plt.show()

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
 u'PlayerID': 0,
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

baseurl = 'http://stats.nba.com/stats/shotchartdetail?'
resp2 = requests.get(baseurl,params=sc_params)
print resp2.url
print resp2.json().keys()

player_shot_df = pd.DataFrame(resp2.json()['resultSets'][0]['rowSet'],columns=resp2.json()['resultSets'][0]['headers'])
print player_shot_df.shape
player_shot_df.head(5)

plt.style.use('seaborn-white')
fig,ax = plt.subplots(figsize=(12, 12))

keepxy = ['LOC_X','LOC_Y']
made_index = (player_shot_df['SHOT_MADE_FLAG']==1)
made_df = player_shot_df.loc[made_index][keepxy]
miss_df = player_shot_df.loc[~made_index][keepxy]

made_df.plot(kind='scatter', x='LOC_X',y='LOC_Y',color='Yellow',
             s=5, alpha=.4, edgecolors='none', zorder=2,ax=ax,
            label='Make')
miss_df.plot(kind='scatter', x='LOC_X',y='LOC_Y',color='Blue',
             s=5, alpha=.4, edgecolors='none', zorder=1,ax=ax,
            label='Miss')
                             
ax.invert_yaxis()

# Set legend to lower right, x-large text, and larger markerscale
leg = ax.legend(loc='lower right',fontsize='x-large',markerscale=5)
labls = leg.get_texts()
[x.set_color('white') for x in labls]

#Read in the half court image
court = plt.imread("img/halfcourt.png")
#The baseline is 4 ft. from front of backboard. front of backboard is 15 in. from center of hoop (the origin).
#Therefore, my image extent needs to go from (4ft.+15in.) = 5.25 ft. so 52.5 ft. in the coordinate system.
img = plt.imshow(court, zorder=0, extent=[-250,250,420,-52.5])

plt.title('All NBA Shots 2015-16',fontsize='x-large')
plt.savefig('img/NBA_shotchart_2015-16.png',dpi=300)
plt.show()

ZAs = player_shot_df.SHOT_ZONE_AREA.unique()
ZRs = player_shot_df.SHOT_ZONE_RANGE.unique()[:-1] #-1 to avoid Backcourt shots

fig,ax = plt.subplots(figsize=(12, 12))
colrs = np.random.random((len(ZAs)*len(ZRs),4))

cnum=0
for za in ZAs:
    for zr in ZRs:
        bin_mask = (player_shot_df['SHOT_ZONE_AREA']==za) & (player_shot_df['SHOT_ZONE_RANGE']==zr)
        if np.sum(bin_mask) > 0:
            cnum+=1
            plt.scatter(x=player_shot_df[bin_mask]['LOC_X'],y=player_shot_df[bin_mask]['LOC_Y'],c=colrs[cnum],
             s=5, alpha=.7, edgecolors='none', zorder=2,label=za+zr)
ax.invert_yaxis()

#Read in the half court image
plt.style.use('grayscale')
court = plt.imread("img/halfcourt.png")
#The baseline is 4 ft. from front of backboard. front of backboard is 15 in. from center of hoop (the origin).
#Therefore, my image extent needs to go from (4ft.+15in.) = 5.25 ft. so 52.5 ft. in the coordinate system.
img = plt.imshow(court, zorder=0, extent=[-250,250,420,-52.5])
plt.title('All NBA Shots Attempts by Shot Zone and Range 2015-16',fontsize='x-large')
plt.savefig('img/NBA_shotzonerange_2015-16.png',dpi=300)
plt.show()

keepfields=['SHOT_TYPE','SHOT_ZONE_AREA','SHOT_ZONE_RANGE','SHOT_ATTEMPTED_FLAG','SHOT_MADE_FLAG']
shot_type_zone_df = player_shot_df[keepfields].groupby(['SHOT_TYPE','SHOT_ZONE_AREA']).sum()
shot_type_zone_df = shot_type_zone_df.assign(SHOT_MADE_PCT = shot_type_zone_df['SHOT_MADE_FLAG']/shot_type_zone_df['SHOT_ATTEMPTED_FLAG'])
shot_type_zone_df

bad_code = player_shot_df[(player_shot_df['SHOT_TYPE']=='2PT Field Goal') & (player_shot_df['SHOT_ZONE_AREA']=='Back Court(BC)')]
bad_code

#It is clearly a 3PT shot (based on distance, x, and y)
player_shot_df.loc[bad_code.index,'SHOT_TYPE']='3PT Field Goal'
#See if any shots with distance > 24 ft. are marked as 2 PT
player_shot_df[(player_shot_df['SHOT_TYPE']=='2PT Field Goal') & (player_shot_df['SHOT_DISTANCE']>=24)]

player_shot_df['ACTION_TYPE'].value_counts()

import re
from collections import OrderedDict

def clean_action_type(text):
    text=re.sub(u'shot',u'Shot',text)
    text=re.sub(u'Driving |Running |Cutting ',u'',text)
    text=re.sub(u'Pullup |Pull-Up |Step Back |Turnaround |Putback |Floating |Finger Roll |Reverse ',u'',text)
    text=re.sub(u'Tip Layup Shot|Tip Dunk Shot',u'Tip',text)
    text=re.sub(u'Alley Oop Dunk Shot|Alley Oop Layup Shot',u'Alley Oop',text)
    text=re.sub(u'Fadeaway Shot|Fadeaway Bank Shot|Fadeaway Bank Jump Shot',u'Fadeaway Jump Shot',text)
    text=re.sub(u'Bank Jump Shot|Jump Bank Shot|Bank Shot|Fadeaway Jump Shot',u'Jump Shot',text)
    text=re.sub(u'Bank Hook Shot|Hook Bank Shot|Hook Jump Shot',u'Hook Shot',text)
    return text

def categorize_action_type(text):
    keywords = OrderedDict()
    keywords["Alley Oop"] = ["Alley Oop"] #alley oop first (so alley oop dunks -> alley oop and not dunk)
    keywords["Tip"] = ["Tip"] #tip next
    keywords["Dunk"] = ["Dunk"]
    keywords["Layup"] = ["Layup"]
    keywords["Hook"] = ["Hook"]
    keywords["Jump Shot"] = ["Jump","Fadeaway"]
    
    keywords_res = OrderedDict()
    for k in keywords:
        pat = "\\b%s\\b" % "\\b|\\b".join(keywords[k])
        keywords_res[k] = re.compile(pat, re.IGNORECASE)
    
    for r in keywords_res:
        if keywords_res[r].search(text): return r
    return "Other"
    
#action_counts = player_shot_df['ACTION_TYPE'].map(lambda x: clean_action_type(x)).value_counts()
action_counts = player_shot_df['ACTION_TYPE'].map(lambda x: categorize_action_type(x)).value_counts()

#plt.style.use('seaborn-white')
plt.style.use('fivethirtyeight')

#action_counts.drop('No Shot').plot(kind='pie',figsize=(8,8),autopct='%.2f',colors=['b','g','r','c','m','y','gray','orange'])
action_counts.drop('Other').plot(kind='pie',figsize=(10,10),autopct='%.2f',colors=['orange','g','r','c','m','y','gray'])

player_shot_df['SHOT_ZONE_BASIC'].value_counts().plot(kind='pie',figsize=(10,10),autopct='%.2f',colors=['orange','g','r','c','m','y','gray'])

player_shot_df[player_shot_df['ACTION_TYPE']=='No Shot']

def game_event_id_video_link(gid,gevid):
    if isinstance(gid,int):
        gid = str(gid)
    if isinstance(gevid,int):
        gevid = str(gevid)
    return 'http://stats.nba.com/cvp.html?GameID='+gid+'&GameEventID='+gevid

for f,row in player_shot_df[player_shot_df['ACTION_TYPE']=='No Shot'].iterrows():
    print game_event_id_video_link(row['GAME_ID'],row['GAME_EVENT_ID'])

keepfields = ['PLAYER_ID','PLAYER_NAME','TEAM_NAME','SHOT_TYPE','SHOT_ZONE_AREA','SHOT_ZONE_RANGE','SHOT_ATTEMPTED_FLAG','SHOT_MADE_FLAG']
shot_zones_df=player_shot_df[keepfields].groupby(['PLAYER_ID','PLAYER_NAME','TEAM_NAME','SHOT_TYPE','SHOT_ZONE_AREA','SHOT_ZONE_RANGE']).sum()
shot_zones_df.head(15)

mask = shot_zones_df['SHOT_ATTEMPTED_FLAG']>=10
shot_zones_df = shot_zones_df.loc[mask]
#now create 'SHOT_MADE_FLAG' column using the assign function
shot_zones_df = shot_zones_df.assign(FG_PCT = shot_zones_df['SHOT_MADE_FLAG']/shot_zones_df['SHOT_ATTEMPTED_FLAG'])
shot_zones_df.head(5)

idx=pd.IndexSlice
print '===== 2PT FGS ====='
print '2PT FGS < 8 ft. ======='
print shot_zones_df.loc[idx[:,:,:,['2PT Field Goal'],['Center(C)'],['Less Than 8 ft.']],idx[['SHOT_MADE_FLAG']]].sort_values(by='SHOT_MADE_FLAG',ascending=False)[:10]
print '\n2PT FGS 8-16 ft. ======='
print shot_zones_df.loc[idx[:,:,:,['2PT Field Goal'],['Right Side(R)'],['8-16 ft.']],idx[['SHOT_MADE_FLAG']]].sort_values(by='SHOT_MADE_FLAG',ascending=False)[:10]
print shot_zones_df.loc[idx[:,:,:,['2PT Field Goal'],['Center(C)'],['8-16 ft.']],idx[['SHOT_MADE_FLAG']]].sort_values(by='SHOT_MADE_FLAG',ascending=False)[:10]
print shot_zones_df.loc[idx[:,:,:,['2PT Field Goal'],['Left Side(L)'],['8-16 ft.']],idx[['SHOT_MADE_FLAG']]].sort_values(by='SHOT_MADE_FLAG',ascending=False)[:10]
print '\n2PT FGS 16-24 ft. ======='
print shot_zones_df.loc[idx[:,:,:,['2PT Field Goal'],['Right Side(R)'],['16-24 ft.']],idx[['SHOT_MADE_FLAG']]].sort_values(by='SHOT_MADE_FLAG',ascending=False)[:10]
print shot_zones_df.loc[idx[:,:,:,['2PT Field Goal'],['Right Side Center(RC)'],['16-24 ft.']],idx[['SHOT_MADE_FLAG']]].sort_values(by='SHOT_MADE_FLAG',ascending=False)[:10]
print shot_zones_df.loc[idx[:,:,:,['2PT Field Goal'],['Center(C)'],['16-24 ft.']],idx[['SHOT_MADE_FLAG']]].sort_values(by='SHOT_MADE_FLAG',ascending=False)[:10]
print shot_zones_df.loc[idx[:,:,:,['2PT Field Goal'],['Left Side Center(LC)'],['16-24 ft.']],idx[['SHOT_MADE_FLAG']]].sort_values(by='SHOT_MADE_FLAG',ascending=False)[:10]
print shot_zones_df.loc[idx[:,:,:,['2PT Field Goal'],['Left Side(L)'],['16-24 ft.']],idx[['SHOT_MADE_FLAG']]].sort_values(by='SHOT_MADE_FLAG',ascending=False)[:10]

print '\n===== 3PT FGS ====='
print shot_zones_df.loc[idx[:,:,:,['3PT Field Goal'],['Right Side(R)'],['24+ ft.']],idx[['SHOT_MADE_FLAG']]].sort_values(by='SHOT_MADE_FLAG',ascending=False)[:10]
print shot_zones_df.loc[idx[:,:,:,['3PT Field Goal'],['Right Side Center(RC)'],['24+ ft.']],idx[['SHOT_MADE_FLAG']]].sort_values(by='SHOT_MADE_FLAG',ascending=False)[:10]
print shot_zones_df.loc[idx[:,:,:,['3PT Field Goal'],['Center(C)'],['24+ ft.']],idx[['SHOT_MADE_FLAG']]].sort_values(by='SHOT_MADE_FLAG',ascending=False)[:10]
print shot_zones_df.loc[idx[:,:,:,['3PT Field Goal'],['Left Side Center(LC)'],['24+ ft.']],idx[['SHOT_MADE_FLAG']]].sort_values(by='SHOT_MADE_FLAG',ascending=False)[:10]
print shot_zones_df.loc[idx[:,:,:,['3PT Field Goal'],['Left Side(L)'],['24+ ft.']],idx[['SHOT_MADE_FLAG']]].sort_values(by='SHOT_MADE_FLAG',ascending=False)[:10]

idx=pd.IndexSlice
print '===== 2PT FGS ====='
print '2PT FGS < 8 ft. ======='
print shot_zones_df.loc[idx[:,:,:,['2PT Field Goal'],['Center(C)'],['Less Than 8 ft.']],idx[['FG_PCT']]].sort_values(by='FG_PCT',ascending=False)[:10]
print '\n2PT FGS 8-16 ft. ======='
print shot_zones_df.loc[idx[:,:,:,['2PT Field Goal'],['Right Side(R)'],['8-16 ft.']],idx[['FG_PCT']]].sort_values(by='FG_PCT',ascending=False)[:10]
print shot_zones_df.loc[idx[:,:,:,['2PT Field Goal'],['Center(C)'],['8-16 ft.']],idx[['FG_PCT']]].sort_values(by='FG_PCT',ascending=False)[:10]
print shot_zones_df.loc[idx[:,:,:,['2PT Field Goal'],['Left Side(L)'],['8-16 ft.']],idx[['FG_PCT']]].sort_values(by='FG_PCT',ascending=False)[:10]
print '\n2PT FGS 16-24 ft. ======='
print shot_zones_df.loc[idx[:,:,:,['2PT Field Goal'],['Right Side(R)'],['16-24 ft.']],idx[['FG_PCT']]].sort_values(by='FG_PCT',ascending=False)[:10]
print shot_zones_df.loc[idx[:,:,:,['2PT Field Goal'],['Right Side Center(RC)'],['16-24 ft.']],idx[['FG_PCT']]].sort_values(by='FG_PCT',ascending=False)[:10]
print shot_zones_df.loc[idx[:,:,:,['2PT Field Goal'],['Center(C)'],['16-24 ft.']],idx[['FG_PCT']]].sort_values(by='FG_PCT',ascending=False)[:10]
print shot_zones_df.loc[idx[:,:,:,['2PT Field Goal'],['Left Side Center(LC)'],['16-24 ft.']],idx[['FG_PCT']]].sort_values(by='FG_PCT',ascending=False)[:10]
print shot_zones_df.loc[idx[:,:,:,['2PT Field Goal'],['Left Side(L)'],['16-24 ft.']],idx[['FG_PCT']]].sort_values(by='FG_PCT',ascending=False)[:10]

print '\n===== 3PT FGS ====='
print shot_zones_df.loc[idx[:,:,:,['3PT Field Goal'],['Right Side(R)'],['24+ ft.']],idx[['FG_PCT']]].sort_values(by='FG_PCT',ascending=False)[:10]
print shot_zones_df.loc[idx[:,:,:,['3PT Field Goal'],['Right Side Center(RC)'],['24+ ft.']],idx[['FG_PCT']]].sort_values(by='FG_PCT',ascending=False)[:10]
print shot_zones_df.loc[idx[:,:,:,['3PT Field Goal'],['Center(C)'],['24+ ft.']],idx[['FG_PCT']]].sort_values(by='FG_PCT',ascending=False)[:10]
print shot_zones_df.loc[idx[:,:,:,['3PT Field Goal'],['Left Side Center(LC)'],['24+ ft.']],idx[['FG_PCT']]].sort_values(by='FG_PCT',ascending=False)[:10]
print shot_zones_df.loc[idx[:,:,:,['3PT Field Goal'],['Left Side(L)'],['24+ ft.']],idx[['FG_PCT']]].sort_values(by='FG_PCT',ascending=False)[:10]

