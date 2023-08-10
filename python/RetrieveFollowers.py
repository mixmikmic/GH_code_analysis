import twitter,csv
import pprint
pp = pprint.PrettyPrinter() 

# Following the step above copy-paste your keys and secrets. 
Consumer_Key = "" # API key
Consumer_Secret = "" # API secret
Access_Token = ""
Access_Token_Secret = ""

api = twitter.Api(consumer_key=Consumer_Key,
                  consumer_secret=Consumer_Secret,
                  access_token_key=Access_Token,
                  access_token_secret=Access_Token_Secret)

sname = 'simpolproject' #replace it with your own Twitter screen_name
followers = api.GetFollowers(screen_name=sname)

for u in followers:
    print(u.screen_name)
    print(u.followers_count)

followers = [u.screen_name for u in followers if u.followers_count > 100]
print(len(followers))

fname = sname + '_followers.csv'
with open(fname, 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    for f in followers:
        writer.writerow([f])

