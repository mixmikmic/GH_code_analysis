import brscraper
scraper = brscraper.BRScraper()

from datetime import datetime
teams = {'CHC': [], 'CHW': []}
for team in teams:
    for year in range(1901, 2016):
        data = scraper.parse_tables("teams/%s/%d-schedule-scores.shtml" % (team, year))
        last_date = None
        for game in data['team_schedule']:
            if game['Date']:
                date = datetime.strptime(game['Date'], '%A, %b %d')
                game['parsed_date'] = date
                game['year'] = year
                last_date = date
                if date.month < 5:
                    teams[team].append(game)
                else:
                    break
            else:
                game['parsed_date'] = None
                game['year'] = year
                print year
                teams[team].append(game)
                

print 'Cubs have %d games, Sox have %d' % (len(teams['CHC']), len(teams['CHW']))

def create_totals():
    return {'CHC': {'wins': 0, 'losses': 0, 'ties': 0}, 'CHW': {'wins': 0, 'losses': 0, 'ties': 0}}

for year in range(1901, 2016):
    totals = create_totals()
    for team in totals.keys():
        missing_games = False
        for game in teams[team]:
            if not game['parsed_date'] and game['year'] == year:
                missing_games = True
            elif game['year'] == year:
                if game['W/L'].lower().startswith('w'):
                    totals[team]['wins'] += 1
                elif game['W/L'].lower().startswith('l'):
                    totals[team]['losses'] += 1
                elif game['W/L'].lower().startswith('t'):
                    totals[team]['ties'] += 1
                else:
                    print 'wtf: %s' % game['W/L']
        denominator = float(totals[team]['wins'] + totals[team]['losses'] + totals[team]['ties'])
        if denominator != 0:
            print '%s (%d): %d-%d-%d %s(.%.3f)' % (
                team,
                year,
                totals[team]['wins'],
                totals[team]['losses'],
                totals[team]['ties'],
                '' if not missing_games else '(missing some) ',
                float(totals[team]['wins'])/denominator
            )
    print '-----'



