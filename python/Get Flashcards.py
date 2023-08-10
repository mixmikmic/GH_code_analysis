import tweepy 
import wget
import os

auth = tweepy.OAuthHandler(consumer_key, consumer_secret) #Fill these in
auth.set_access_token(access_token, access_token_secret)  #Fill these in

api = tweepy.API(auth)

#Get 200 of Chris' tweet
tweets = api.user_timeline(screen_name = 'chrisalbon', 
                           count = 200, 
                           include_rts = False, 
                           excludereplies = True)

#200 isn't enough.  Keep getting tweets until we can't get anymore

last_id = tweets[-1].id
 
while (True):
    more_tweets = api.user_timeline(screen_name='chrisalbon',
                                count=200,
                                include_rts=False,
                                exclude_replies=True,
                                max_id=last_id-1)
                                    
    # There are no more tweets
    if (len(more_tweets) == 0):
          break
    else:
          last_id = more_tweets[-1].id-1
          tweets = tweets + more_tweets
        
        
#Filter by those containing #machinelearningflashcards 
card_tweets = [j for j in tweets if '#machinelearningflashcards' in j.text]


media_files = set()
for status in card_tweets:
    media = status.entities.get('media', [])
    if(len(media) > 0 and media[0]['type']=='photo' ): #if tweet has media and media is photo
        media_files.add(media[0]['media_url']) #get me the url

os.makedirs('ML Cards') #make a directory to store the photos in

for media_file in media_files:
    wget.download(media_file, out = 'ML Cards') #get the photos!



