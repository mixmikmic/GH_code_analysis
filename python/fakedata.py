import datetime
from os import chdir
import csv

def daterange(start, stop, step):
    while start < stop:
        yield start
        start += step

#Generate fake TMCs
import string
import random
def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

tmcs = []
for i in range (3100):
    tmcs.append(id_generator(9))



#WARNING, TREAD LIGHTLY THIS WILL GENERATE 12 GIGS of DATA
with open('fakedata.csv', 'w',newline='') as fp:
    a = csv.writer(fp, delimiter=',')
    step = datetime.timedelta(seconds=60)
    starttime = datetime.datetime(2012,7,1,0,0,0)
    stoptime = datetime.datetime(2012,8,1,0,0)
    for tx in daterange(starttime,stoptime,step):
        for tmc in tmcs:
            speed = random.randint(1,120)
            score = random.randint(0,30)
            a.writerow([tx,tmc,speed,score])
        



