#import the necessary libraries
import csv, requests, datetime, time
import simplejson as json
import pandas as pd
from twython import Twython

#change this to the SOCRATA portal you want to target, don't forget the trailing slash
targeturl ='http://chhs.data.ca.gov/'

#build data.json url string accaccording to SOCRATA's convention
r=requests.get(targeturl+"api/dcat.json")

#parse the json response into a dictionary named j, coincidentally j's KVPs are also dictionaries
j=r.json()

#stop if something went wrong
if r.status_code==200: print("\nsuccessfully fetched json data, http return code 200")
else: sys.exit()

enter_todays_date='2017-04-24'

memory=[]
for i in j:
    if len(i['identifier']) == 9:
        yr=int(i['created'][:4])
        mo=int(i['created'][5:7])
        dy=int(i['created'][8:10])
        created=datetime.date(yr,mo,dy)

        yr=int(i['modified'][:4])
        mo=int(i['modified'][5:7])
        dy=int(i['modified'][8:10])
        modified=datetime.date(yr,mo,dy)
        
#         delta=today-datetime.date(2017,4,2)
#         print (delta.days)
        
        insert=[i['identifier'],created,modified.isoformat(),enter_todays_date]
        memory.append(insert)
df=pd.DataFrame(data=memory,columns=['id','created','modified','last_tweeted'])
df=df.set_index('id')
print (df)
df.to_csv('memory.csv')

#authenticate with your own twitter application tokens below
twitter = Twython(APP_KEY, APP_SECRET, OAUTH_TOKEN, OAUTH_TOKEN_SECRET)

#function that truncates long titles and sends the tweet
def tweet_it(identifier,title):
    title=title[:89] #if title is too long, truncate it to fit, will require adj. if wording is changed
    x="Updated data \""+title+"\" "+targeturl+"d/"+identifier
    print (x), "debug tweet sent" #for debugging
    twitter.update_status(status=x) #send the tweet
    time.sleep(2) #wait 2 seconds between tweets, this can be adjusted

#declare global 'today' variable for data munging use
today=datetime.datetime.today()

#declare a threshold below which tweets will not be REPEATED
#ex. if a dataset is modified EVERY DAY, threshold=7 will ensure that
#dataset is only tweeted once every 7 days
threshold=7

#read in pre-existing memory file, if one does not exist this will cause an error
#create a memory file by using the code included
memory=pd.read_csv('memory.csv',index_col=0)
print (memory) #debug

for i in j:
    if len(i['identifier']) == 9:
        if i['identifier'] in memory.index.values: #known to us
            c_str=memory.loc[i['identifier']]['created']
            m_str=memory.loc[i['identifier']]['modified']
            l_str=memory.loc[i['identifier']]['last_tweeted']
            c_dt=datetime.datetime.strptime(c_str,'%Y-%m-%d')
            m_dt=datetime.datetime.strptime(m_str,'%Y-%m-%d')
            l_dt=datetime.datetime.strptime(l_str,'%Y-%m-%d')
            delta=(today-l_dt).days>threshold
            valid=(today-m_dt).days<=1
            # print (delta, valid) #debug
            
            if delta and valid:
                print ("valid update")
                tweet_it(i['identifier'],i['title'])
                memory.loc[i['identifier']]['modified']=i['modified'] # update the record's modified date
                memory.loc[i['identifier']]['last_tweeted']=today.strftime('%Y-%m-%d') # update the record's last_tweeted date
            else:
                print ("known but not changed or not old enough")
                memory.loc[i['identifier']]['modified']=i['modified'] # update the record's modified date
        else:
            try:
                print (i['identifier']+"is new, not known to memory")
                tweet_it(i['identifier'],i['title'])
                new_record=pd.DataFrame([[i['identifier'],i['created'],i['modified'],today.strftime('%Y-%m-%d')]], columns=['id','created','modified','last_tweeted'])
                new_record=new_record.set_index('id')
                memory=memory.append(new_record)
                print (new_record)
            except:
                pass
# print (memory)
# print("done")

#create updated memory file after today's changes
memory.to_csv('memory.csv')

