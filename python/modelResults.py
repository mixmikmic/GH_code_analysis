import models
import sims
import teams
import collections
import pandas as pd
import numpy as np
from IPython.display import Image
from IPython.display import Image
from IPython.display import display
from IPython.core.display import HTML
import base64

with open('vis/JazzWeibull.png', "rb") as image_file:
    im1 = 'data:image/png;base64,' + base64.b64encode(image_file.read())

with open("vis/WizardsWeibull.png", "rb") as image_file:
    im2 = 'data:image/png;base64,' + base64.b64encode(image_file.read())

s = """<table>
<tr>
<th><img src="%s"/></th>
<th><img src="%s"/></th>
</tr></table>"""%(im1, im2)
t=HTML(s)
display(t)

m = models.getTeamTrans("Warriors", "off", None)
d = pd.DataFrame(m, columns=["dummy", "dReb","Made 2", "Made 3", "turnover", "steal", "ft(1of1)", "ft(2of2)", "ft(3of3)", "end Qtr"], index=["dummy", "dReb","Made 2", "Made 3", "turnover", "steal", "ft(1of1)", "ft(2of2)", "ft(3of3)", "end Qtr"])
d

print(sims.simGame("Warriors", "Cavaliers"))
print(sims.simGame("Cavaliers", "Warriors"))
print(sims.simGame("Spurs", "76ers"))
print(sims.simGame("76ers", "Spurs"))
print(sims.simGame("Spurs", "Warriors"))
print(sims.simGame("Warriors", "Spurs"))

tms = teams.getAllTeams()
wins = collections.defaultdict(int)

for t1 in tms:
    tm1 = tms[t1]
    for t2 in tms:
        tm2 = tms[t2]
        if tm1['name'] != tm2['name']:
            #print(tm1['name']+ " vs "+tm2['name'])
            res = sims.simGame(tm1['name'], tm2['name'])
            if res[0] > res[1]:
                wins[tm1['loc']+' '+tm1['name']] += 1
            else:
                wins[tm2['loc']+' '+tm2['name']] += 1
            #print(res)

wins = pd.DataFrame.from_dict(wins, orient='index')
wins.columns=['w']
wins.sort('w', ascending=False)


print(wins.sort('w', ascending=False), )


with open('nbastandings.png', "rb") as image_file:
    im1 = 'data:image/png;base64,' + base64.b64encode(image_file.read())

with open("bpi.png", "rb") as image_file:
    im2 = 'data:image/png;base64,' + base64.b64encode(image_file.read())

s = """<table>
<tr>
<th><img src="%s"/></th>
<th><img src="%s"/></th>
</tr></table>"""%(im1, im2)
t=HTML(s)
display(t)



