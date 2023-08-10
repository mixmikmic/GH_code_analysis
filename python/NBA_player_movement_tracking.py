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

IFrame('http://stats.nba.com/cvp.html?GameID=0021500051&GameEventID=383',width=900,height=400)

IFrame('http://stats.nba.com/movement/#!/?GameID=0021500051&GameEventID=383',width=700,height=400)

url = 'http://stats.nba.com/stats/locations_getmoments/?eventid=383&gameid=0021500051'
response = requests.get(url)
response.json().keys()

home = response.json()["home"]
visitor = response.json()["visitor"]
moments = response.json()["moments"]
print response.json()["gamedate"]
print response.json()["gameid"]

home

len(moments)

moments[0]

moments[0][0:5]

moments[0][5][0]

ball_zdat = []
for m in moments:
    ball_zdat.append(m[5][0][4])
fig = plt.figure(figsize=(8, 6))
plt.hist(ball_zdat,bins=range(1,20))
plt.title('Histogram of Z values of basketball')
plt.xlabel('Z values')
plt.ylabel('Frequency')
plt.show()

plt.style.use('ggplot')
fig,ax = plt.subplots(figsize=(8, 6))
#scat = plt.scatter(y=ball_zdat,x=np.arange(0,len(ball_zdat))*.04,c=ball_zdat,cmap=plt.cm.RdBu,s=100)
scat = plt.scatter(y=ball_zdat,x=np.arange(0,len(ball_zdat))*.04,s=100)

court = plt.hlines(0,0,len(ball_zdat)*.04,colors='k')
hoop = plt.hlines(10,0,len(ball_zdat)*.04,colors='orange')
plt.autoscale(tight=True)
plt.tight_layout()
plt.ylabel('Ball height (feet)')
plt.xlabel('Time from start of event (seconds)')
plt.legend((hoop,court),('hoop rim', 'court'),fontsize=12)
#cbar = fig.colorbar(scat,orientation='vertical',fraction=0.05)
plt.title('Ball height over time',fontsize='x-large')
plt.savefig('img/NBA_ballheight.png',dpi=300)
plt.show()

ball_xdat = []
ball_ydat = []
for m in moments:
    ball_xdat.append(m[5][0][2])
    ball_ydat.append(m[5][0][3])

plt.style.use('seaborn-white')
fig = plt.figure(figsize=(15, 11.5))

scat = plt.scatter(x=ball_xdat,y=ball_ydat,c=ball_zdat,
    cmap=plt.cm.RdBu, s=300, zorder=1)
plt.style.use('grayscale')
court = plt.imread("img/fullcourt.png")
img = plt.imshow(court, zorder=0, extent=[0,94,50,0])
plt.style.use('seaborn-white')
cbar = fig.colorbar(scat,orientation='vertical',fraction=0.025)
cbar.ax.invert_xaxis()
cbar.ax.set_ylabel('Ball height (ft.) above court')
# xaxis is 0-94 ft. yaxis 50-0 feet (inverted)
import matplotlib.font_manager as fm
# zorder should be less than zorder of scatter, so court is drawn first.
annotation_font = fm.FontProperties(family='Bitstream Vera Sans',style='normal',size=10,weight='normal',stretch='normal')
#scat.annotate('[START] Connelly drives baseline',
#              xy = ball_xdat

#
#plt.imshow(court, zorder=0, extent=[0,94,50,0])

#plt.xlim(0,94)
plt.title('Ball movement',fontsize='x-large')
plt.savefig('img/Ball_movement.png',dpi=300)
plt.show()

moment_data = []
for m in moments:
    for p in m[5]:
        p+=[moments.index(m), m[2],m[3]]
        moment_data.append(p)

moment_data[0]

pm_df = pd.DataFrame(data=moment_data,columns=['teamID','playerID','xloc','yloc','zloc','moment','gametime','shottime'])
pm_df.head(15)

def players_to_df(players):
    players_reform=[[rec[arg] for arg in players["players"][0].keys()] for rec in players["players"]]
    df=pd.DataFrame(players_reform,columns=players["players"][0].keys())
    df['team']=players["abbreviation"]
    return df

players_df = pd.concat([players_to_df(home),players_to_df(visitor)])
players_df

pm_merge_df = pm_df.merge(players_df,how='left',left_on='playerID',right_on='playerid')
pm_merge_df.head(11)

from matplotlib.patches import Rectangle, Circle, Arc
from matplotlib.collections import PatchCollection
import matplotlib as mpl
import copy
clr='red'

def left_court_patches(clr='red'):
    # creates the patches for the left hand side of an nba court
    # returns: list of patches
    rim = Circle((5.25,-25),radius=.75, color=clr, zorder=0, lw=.5, fill=False)
    bkbrd = Rectangle((4,-28),width=0, height=6, color=clr, zorder=0,fill=False,lw=1)
    ra_arc = Arc((5.25,-25),8,8,theta1=270,theta2=90,color=clr,zorder=9,fill=False,lw=1)
    ra_t = Rectangle((4,-21),width=1.25,height=0,color=clr, zorder=0,fill=False,lw=1)
    ra_b = Rectangle((4,-29),width=1.25,height=0,color=clr, zorder=0,fill=False,lw=1)
    ft = Rectangle((0,-33),width=19,height=16,color=clr, zorder=0,fill=False,lw=1) #*16ft. outside
    lane = Rectangle((0,-31),width=19,height=12,color=clr,zorder=0,fill=False,lw=1)#*12ft. outside
    ft_ia = Arc((19,-25),12,12,theta1=90,theta2=270,color=clr,zorder=9,fill=False,lw=1,linestyle='--')
    ft_oa = Arc((19,-25),12,12,theta1=270,theta2=90,color=clr,zorder=9,fill=False,lw=1)
    three_t = Rectangle((0,-3),width=14,height=0,color=clr,zorder=0,fill=False,lw=1)
    three_b = Rectangle((0,-47),width=14,height=0,color=clr,zorder=0,fill=False,lw=1)
    three_arc = Arc((5.25,-25),23.75*2,23.75*2,theta1=292,theta2=68,color=clr,zorder=0,fill=False,lw=1)
    return [rim,bkbrd,ra_arc,ra_t,ra_b,ft,lane,ft_ia,ft_oa,three_t,three_b,three_arc]

def draw_full_court(ax=None,clr='red'):
    # Draws full court onto axis
    # returns: axis
    if ax is None:
        ax = plt.gca()
        
    #full court components 
    fullcourt = Rectangle((0,-50), width=94, height=50, color=clr, zorder=0,fill=False,lw=1) #base and side lines
    midcourt = Rectangle((47,-50), width=0, height=50, color = clr, zorder=0,fill=False,lw=1) #half-court line
    rest_circ = Circle((47,-25),radius=2,color=clr,zorder=0,lw=1,fill=False) #restraining circle
    cent_circ = Circle((47,-25),radius=6,color=clr,zorder=0,lw=1,fill=False) #center circle
    patch_list = [fullcourt,midcourt,rest_circ,cent_circ]
    for patch in patch_list:
        ax.add_patch(patch)
    
    #left-hand side
    left_patches = left_court_patches(clr=clr)
    for patch in left_patches:
        ax.add_patch(patch)

    #right-hand side transform patches by rotating pi radians around center point
    right_patches = left_court_patches(clr=clr)
    tr = mpl.transforms.Affine2D().rotate_around(47,-25,np.pi) + ax.transData
    for patch in right_patches:
        patch.set_transform(tr)
        ax.add_patch(patch)
    return ax

plt.style.use('seaborn-white')
fig,ax = plt.subplots(figsize=(9.4,5.0))
draw_full_court()

plt.xlim(0,94)
plt.ylim(-50,0)
plt.show()

def plot_moment(df,moment=0,tc=['cyan','yellow'],ax=None):
    if df is None:
        raise ValueError("Input 'df' to plot_moment is None")
    if ax is None:
        ax=plt.gca()
        
    binary_mask = (df['moment']==moment)
    moment_df = df[binary_mask]
    teams = moment_df.team.dropna().unique().tolist()
    for t in teams:
        team_mask = (moment_df['team']==t)
        plt.scatter(x=moment_df[team_mask]['xloc'],y=-moment_df[team_mask]['yloc'],
                    c=tc[teams.index(t)], s=200, zorder=1, label=t, edgecolor=tc[teams.index(t)])
        for j,x,y in zip(moment_df[team_mask]['jersey'], moment_df[team_mask]['xloc'], moment_df[team_mask]['yloc']):
            plt.annotate(j,xy=(x,-y),xytext=(0,0),textcoords='offset points',
                         zorder=2, ha='center',va='center',snap=False)
        #TODO(maybe): just plot annotations with circular bboxes, no scatter
    ball_mask=(moment_df['playerID']==-1)
    plt.scatter(x=moment_df[ball_mask]['xloc'],y=-moment_df[ball_mask]['yloc'],
                s=100,zorder=3,c='orange',edgecolor='orange',label='ball')

plt.style.use('seaborn-white')
fig,ax = plt.subplots(figsize=(9.4,5.0))
draw_full_court(ax=ax,clr='gray')
plot_moment(pm_merge_df)
plt.legend(loc='lower right')
plt.xlim(0,94)
plt.ylim(-50,0)
plt.title('Player and Ball position at t=0',fontsize='x-large')
plt.savefig('img/Player_ball_positions.png',dpi=300)
plt.show()



