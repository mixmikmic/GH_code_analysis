import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
get_ipython().magic('matplotlib inline')
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import shotpredictor

import random

import json
from pprint import pprint

def getGameJSON(path):
    with open(path) as data_file:    
        data = json.load(data_file)
    return data
        
data = getGameJSON('GSWvsCLE.json')

headers = ["team_id", "player_id", "x_loc", "y_loc", 
           "radius"]

def playerDFtoList(frame):
    xList = frame['x_loc'].tolist()
    yList = frame['y_loc'].tolist()
    bothList = []
    for i in range(len(xList)):
        bothList.append([ xList[i], yList[i] ])
    return np.array(bothList)

def getMoments(data):
    player_moments = []
    for event in data['events']:
        for moment in event['moments']:
            for player in moment[5]:
                player_moments.append(player)

    df = pd.DataFrame(player_moments, columns=headers)
    return df

min_5 = 25*60*6*11
# index = np.arange(0, min_5/11, 1)
index = np.arange(0, 6812, 1)

def dist(data, ball):
    data["distanceToBall"] = np.sqrt((data["x_loc"] -ball["x_loc"])**2+ (data["y_loc"] -ball["y_loc"])**2)

def getDistBetween(a,b):
    return np.sqrt((a["x_loc"] - b["x_loc"])**2 + (a["y_loc"] - b["y_loc"])**2)
    
def hasBall(data):
    data["hasBall"] = data["distanceToBall"].apply(lambda x: 1 if x < 2.5 else 0)
 
def defDist(l):
    # Order: [steph, klay, bogut, green, barnes, lebron, smith, love, irving, mozgov]
    for player in l:
        if (player['team_id'][0] == 1610612739):
            # CLE player, get dist to all GSW players
            player['distToCurry'] = getDistBetween(player, l[0])
            player['distToThompson'] = getDistBetween(player, l[1])
            player['distToBogut'] = getDistBetween(player, l[2])
            player['distToGreen'] = getDistBetween(player, l[3])
            player['distToBarnes'] = getDistBetween(player, l[4])
            player['distToNearestDef'] = player.loc[:, ['distToCurry', 'distToThompson', 'distToBogut', 'distToGreen', 'distToBarnes']].min(axis=1)
            # Drop columns 7-11
            player.drop(player.columns[[5, 6, 7, 8, 9]], axis=1, inplace=True)            
        elif (player['team_id'][0] == 1610612744):
            # GSW player, get dist to all CLE players
            player['distToJames'] = getDistBetween(player, l[5])
            player['distToSmith'] = getDistBetween(player, l[6])
            player['distToLove'] = getDistBetween(player, l[7])
            player['distToIrving'] = getDistBetween(player, l[8])
            player['distToMozgov'] = getDistBetween(player, l[9])
            player['distToNearestDef'] = player.loc[:, ['distToJames', 'distToSmith', 'distToLove', 'distToIrving', 'distToMozgov']].min(axis=1)
            # Drop columns 7-11
            player.drop(player.columns[[5, 6, 7, 8, 9]], axis=1, inplace=True)


def transform(l, ball, skipRanges):
    ball.reset_index(drop=True, inplace =True)    
    result = []
    for r in skipRanges:
        ball.drop(ball.index[r[0]:r[1]], inplace=True)
    ball.reset_index(drop=True, inplace =True)    
    print "reset ball"
    
    for player in l:
        player.reset_index(drop=True, inplace =True)
        for r in skipRanges:
            player.drop(player.index[r[0]:r[1]], inplace=True)
        player.reset_index(drop=True, inplace =True)

        
    defDist(l)
    for player in l:
        dist(player, ball)
        hasBall(player)
        result.append(player[(player.hasBall == 1)][["player_id", "team_id", "x_loc", "y_loc", "distToNearestDef"]])
        
    df_res = pd.concat(result)
    pos = df_res.sort_index()
    idx = np.unique(pos.index, return_index=True)[1]
    pos= pos.iloc[idx]
    print "reindexing"
    pos = pos.reindex(index, fill_value=0)
    return pos

def removeRepeats(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]        

# Get moments from the zeroth event
player_moments = []
endsOfEvents = []
counter = -1
for event in data['events']:
    for moment in event['moments']:
        counter += 1
        for player in moment[5]:
            player_moments.append(player)
    endsOfEvents.append(counter)

df = pd.DataFrame(player_moments, columns=headers)
df = df.head(min_5)
endsOfEvents = removeRepeats(endsOfEvents)

# Get specific player's movements
ball = df[df.player_id==-1]

steph_df = df[df.player_id==201939]
klay_df = df[df.player_id==202691]
bogut_df = df[df.player_id==101106]
green_df = df[df.player_id==203110]
barnes_df = df[df.player_id==203084]

lebron_df = df[df.player_id==2544]
smith_df = df[df.player_id==2747]
love_df = df[df.player_id==201567]
irving_df = df[df.player_id==202681]
mozgov_df = df[df.player_id==202389]

ballA = playerDFtoList(ball)

# Get the indexes that overlap
skipRanges = []
# Only get the first 17 indices b/c we're only doing the first 5 minutes of the game
for eventEndIndex in endsOfEvents[:18]:
    lastLocation = ballA[eventEndIndex]
#     print '       Event end at '+str(eventEndIndex)+':',lastLocation
    for i in range(eventEndIndex+1,len(ballA[eventEndIndex+1:])):
        if (ballA[i] == lastLocation).all():
#             print 'matched location at '+str(i)+':',ballA[i]
            skipRanges.append([eventEndIndex, i])
            break
print 'Ranges to skip:',skipRanges

l = [steph_df,klay_df,bogut_df,green_df,barnes_df,lebron_df,smith_df,love_df,irving_df,mozgov_df]

pos = transform(l, ball, skipRanges)
a_pos = playerDFtoList(pos)

def distance(ball):
    basket1 = [5.32, 24.8]
    basket2 = [88.55, 24.8]
    halfcourt = 47.0
    
    ball['cavsHoop'] = np.sqrt((basket1[0] - ball['x_loc'])**2 + (basket1[1] - ball['y_loc'])**2)
    ball['warriorsHoop'] = np.sqrt((basket2[0] - ball['x_loc'])**2 + (basket2[1] - ball['y_loc'])**2)
    
distance(ball)

pos = pd.concat([pos, ball[["cavsHoop", "warriorsHoop"]]], axis=1)

dic = {201939 : "stephen curry",
202691 : "klay thompson",
101106 : "andrew bogut",
203110 : "draymond green",
203084 : "harrison barnes",
2544 : "lebron james",
2747 : "j.r. smith",
201567 : "kevin love",
202681 : "kyrie irving",
202389 : "timofey mozgov"}
team = {"1610612744":"Warriors",
"1610612739": "Cavaliers"}

guards_100 = [
    "James Harden","Damian Lillard","Chris Paul","John Wall","Eric Bledsoe","Joe Johnson","Kyrie Irving","Monta Ellis",
     "Tyreke Evans","Ben McLemore","Ty Lawson","Goran Dragic","Stephen Curry","Kentavious Caldwell-Pope","Victor Oladipo",
     "Jimmy Butler","Arron Afflalo","Elfrid Payton","Klay Thompson","Avery Bradley","Kyle Korver","Kyle Lowry",
     "JJ Redick","Mario Chalmers","Courtney Lee","Gerald Henderson","Danny Green","Russell Westbrook","Trey Burke",
     "Reggie Jackson","Evan Turner","Jarrett Jack","Jeff Teague","Mike Conley","Dion Waiters","Michael Carter-Williams",
     "Kemba Walker","Deron Williams","Bradley Beal","DeMar DeRozan","J.R. Smith","Andre Iguodala","Brandon Knight",
     "Wesley Matthews","Eric Gordon","Rajon Rondo","Lou Williams"]
    
#     ,"Greivis Vasquez","Mo Williams","Dwyane Wade","D.J. Augustin",
#      "Tony Parker","Jeremy Lin","Zach LaVine","Aaron Brooks","Rodney Stuckey","Shane Larkin","Bojan Bogdanovic","CJ Miles",
#      "Norris Cole","Dante Exum","Marcus Smart","Anthony Morrow","Quincy Pondexter","Hollis Thompson","Alan Anderson",
#      "Patrick Beverley","Isaiah Thomas","Jerryd Bayless","Jamal Crawford","O.J. Mayo","Devin Harris","Tim Hardaway",
#      "Wayne Ellington","Evan Fournier","Tony Allen","Jason Terry","Kirk Hinrich","Manu Ginobili","Lance Stephenson",
#      "Darren Collison","Austin Rivers","Iman Shumpert","Derrick Rose","Steve Blake","Dennis Schroder","Rasual Butler",
#      "Beno Udrih","Jordan Clarkson","Shaun Livingston","Jodie Meeks","Langston Galloway","Gerald Green","Cory Joseph",
#      "Ray McCallum","CJ Watson","Tony Snell","Jameer Nelson","Marco Belinelli","Matthew Dellavedova"
# ]

forwards_100 = [
    "Andrew Wiggins","Trevor Ariza","Pau Gasol","Gordon Hayward","Markieff Morris","Giannis Antetokounmpo","Kevin Love",
    "LaMarcus Aldridge","LeBron James","Draymond Green","Wilson Chandler","Anthony Davis","Jeff Green","Thaddeus Young",
    "Luol Deng","Rudy Gay","Paul Millsap","PJ Tucker","Solomon Hill","Nicolas Batum","Khris Middleton","Tobias Harris",
    "Blake Griffin","Harrison Barnes","Al Horford","Nerlens Noel","Zach Randolph","Josh Smith","Dirk Nowitzki",
    "Derrick Favors","Matt Barnes","Wesley Johnson","Tim Duncan","Tristan Thompson","DeMarre Carroll","Chandler Parsons",
    "Patrick Patterson","Serge Ibaka","Terrence Ross","Corey Brewer","Kenneth Faried","Marcus Morris","Donatas Motiejunas",
    "Marvin Williams","Kawhi Leonard","Jason Thompson","Boris Diaw","Amir Johnson","Robert Covington","Brandon Bass",
    "Paul Pierce","David West","Channing Frye","Ed Davis"]
# ,"Mike Dunleavy","Jason Smith","Mason Plumlee",
#     "Kyle Singler","Tyler Zeller","Jared Dudley","Taj Gibson","Ryan Anderson","Joe Ingles","Luis Scola",
#     "Nikola Mirotic","Dante Cunningham","Jae Crowder","Caron Butler","Michael Kidd-Gilchrist","Jared Sullinger","Trevor Booker",
#     "Chris Bosh","Lance Thomas","Cody Zeller","Derrick Williams","Brandan Wright","Otto Porter","Anthony Tolliver",
#     "Carmelo Anthony","Danilo Gallinari","Kelly Olynyk","Omri Casspi","JJ Hickson","Tayshaun Prince","Jerami Grant",
#     "James Johnson","Al-Farouq Aminu","Kris Humphries","Ersan Ilyasova","Quincy Acy","Chase Budinger","Amar'e Stoudemire",
#     "Richard Jefferson","Kevin Seraphin","Ryan Kelly","Jonas Jerebko","John Henson","Carl Landry"
# ]

centers_100 = [
    "DeAndre Jordan","Marc Gasol","Nikola Vucevic","Andre Drummond","Marcin Gortat","Tyson Chandler","Gorgui Dieng",
    "Rudy Gobert","Greg Monroe","Enes Kanter","Brook Lopez","Jonas Valanciunas","Joakim Noah","Timofey Mozgov",
    "DeMarcus Cousins","Al Jefferson","Omer Asik","Roy Hibbert","Jordan Hill","Steven Adams","Zaza Pachulia",
    "Robin Lopez","Andrew Bogut","Alex Len","Henry Sims","Chris Kaman","Kosta Koufos","Spencer Hawes","Bismack Biyombo",
    "Dwight Howard","Marreese Speights","Tarik Black","Miles Plumlee","Kendrick Perkins","Ian Mahinmi","Hassan Whiteside",
    "Robert Sacre","Aron Baynes","Jusuf Nurkic","Cole Aldrich","Alexis Ajinca","Meyers Leonard","Dewayne Dedmon",
    "Kyle O'Quinn","Nikola Pekovic","Justin Hamilton","Samuel Dalembert","Festus Ezeli","Ryan Hollins","Joel Anthony",
    "Jerome Jordan","Greg Smith","Jeff Withey","JaVale McGee","Bernard James","Earl Barron","Nazr Mohammed","Clint Capela"
]

def shot_dist(dist):
    if dist < 8:
        return "less than 8"
    elif dist < 16:
        return "8-16"
    elif dist < 24:
        return "16-24"
    else:
        return "24+"
    
def shot_decide(dist):
    if dist == 0:
        return 0
    elif dist < 8:
        if random.randint(0, 4) == 0:
            return "else"
        else:
            return "layup"
    else:
        return "jump"
    
def addParameters(data):
    data = data.copy()
    a = data["cavsHoop"][data["team_id"] == 1610612739]
    b= data["warriorsHoop"][data["team_id"] == 1610612744]
#     add distanceToBasket
    data["distanceToBasket"] = pd.concat([a,b]).reindex(index, fill_value=0)
    
    shot_dist_c = pd.get_dummies(data["distanceToBasket"].apply(shot_dist))
    
    shot_decide_c = pd.get_dummies(data["distanceToBasket"].apply(shot_decide))
    
    data["player"] = data["player_id"].apply(lambda x: 0 if x == 0 else dic[x])
    
    
    
#     return pd.concat([data, shot_dist_c,shot_decide_c], axis =1)
    return pd.concat([data,shot_decide_c], axis =1)
#     return shot_dist_c
    

final = addParameters(pos)

curry, thompson, bogut, green, barnes, james, smith, love, irving, timofey= 0,0,0,0,0,0,0,0,0,0
model_dic = {201939 : curry,
202691 : thompson,
101106 : bogut,
203110 : green,
203084 : barnes,
2544 : james,
2747 : smith,
201567 : love,
202681 : irving,
202389 :timofey }

guards_model, guards_pred = shotpredictor.large_model(guards_100)

forwards_model, forwards_pred = shotpredictor.large_model(forwards_100)

centers_model, centers_pred = shotpredictor.large_model(centers_100)

large_dic = {201939 : [guards_model, guards_pred],
202691 : [guards_model,guards_pred],
101106 : [centers_model, centers_guard],
203110 : [forwards_model,forwards_pred],
203084 : [forwards_model,forwards_pred],
2544 : [forwards_model,forwards_pred],
2747 : [guards_model,guards_pred],
201567 : [forwards_model,forwards_pred], 
202681 : [guards_model,guards_pred],
202389 : [centers_model, centers_guard] }

for player_id in dic:
    print dic[player_id]
    model_dic[player_id] = shotpredictor.predictor(dic[player_id], "2014")

model_dic[201939].predict_proba([24,6,0,1,0])

def buildOneHot(player, pred):
    a = []
    for p in pred:
        if p == player:
            a.append(1)
        else:
            a.append(0)
    return a
    

# pred = final[['distanceToBasket', 'else', 'jump', 'layup']]
# pct = [] 
# for i,j in pred.iterrows():
#     p_id = final["player_id"][i]
#     if p_id == 0:
#         pct.append(0)
#     else:
#         p_array = buidOneHot(dic[p_id], model_dic[p_id])
#         pct.append(model_dic[p_id].predict_proba(p_array + j.tolist())[0][1])
# final["pct"] = pct

pred = final[['distanceToBasket', 'distToNearestDef', 'else', 'jump', 'layup']]
# pred = final[['distanceToBasket', 'else', 'jump', 'layup']]
pct = [] 
for i,j in pred.iterrows():
    p_id = final["player_id"][i]
    if p_id == 0:
        pct.append(0)
    else:
        pct.append(model_dic[p_id].predict_proba(j.tolist())[0][1])
final["pct"] = pct

final["pct"]

# convert them to numpy arrays
ballA = playerDFtoList(ball)

stephA = playerDFtoList(steph_df)
klayA = playerDFtoList(klay_df)
bogutA = playerDFtoList(bogut_df)
greenA = playerDFtoList(green_df)
barnesA = playerDFtoList(barnes_df)

lebronA = playerDFtoList(lebron_df)
smithA = playerDFtoList(smith_df)
loveA = playerDFtoList(love_df)
irvingA = playerDFtoList(irving_df)
mozgovA = playerDFtoList(mozgov_df)

#Get ball radiuses
radii = ball['radius'].tolist()

ex = {}
ex["Ball"] = ballA.tolist()

ex["Curry"] = stephA.tolist()
ex["Thompson"] = klayA.tolist()
ex["Bogut"] = bogutA.tolist()
ex["Green"] = greenA.tolist()
ex["Barnes"] = barnesA.tolist()


ex["James"] = lebronA.tolist()
ex["Smith"] = smithA.tolist()
ex["Love"] = loveA.tolist()
ex["Irving"] = irvingA.tolist()
ex["Mozgov"] = mozgovA.tolist()

ex["radius"] = radii

ex["pos"] = a_pos.tolist()

ex["pct"] = final["pct"].tolist()

# print ex['steph'][148:152]
# print ex['steph'][299:301]

with open("./website/public/dump.json", "w") as outfile:
    json.dump(ex, outfile)

