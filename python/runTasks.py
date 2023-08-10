import urllib2
import json
import sims

def getOdds(*args):
    #args[0] if exists is date in YYYYMMDD
    #odds are only made day of or right before, so will only work with no input or datestring of tmrw
    url = "http://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
    if len(args) > 0:
        url = url + "?dates=" + args[0]
    #print(url)
    data = json.loads(urllib2.urlopen(url).read())
    games = data['events']
    todays = []
    for game in games:
        game = game['competitions'][0]

        #make sure the game is pre, otherwise theres no odds
        status = game['status']['type']['state']
        if status != 'pre':
            continue

        #get teams
        home = game['competitors'][0]['team']['name']
        away = game['competitors'][1]['team']['name']

        #get odds
        if 'odds' in game:
            line = game['odds'][0]['details']
            ou = game['odds'][0]['overUnder']
            todays.append((away, home, line, ou))
        
    return todays

def getPredictions(*args):
    games = getOdds(*args)
    #print(games)
    allSims = []
    for game in games:
        sg = sims.simGame(game[0], game[1])
        allSims.append(game + sg[0:2])
    return allSims

getPredictions()



