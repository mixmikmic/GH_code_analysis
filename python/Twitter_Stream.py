get_ipython().magic('pylab')
import tweepy
import json
import pysd
import threading
from matplotlib import animation

from _twitter_credentials import *

model = pysd.read_vensim('../../models/Twitter/Twitter.mdl')
model.set_components({'displacement_timescale':30})

counter = 0

class TweetListener(tweepy.StreamListener):
    def on_data(self, data):
        global counter
        counter += 1
        
        # Twitter returns data in JSON format - we need to decode it first
        decoded = json.loads(data)

        # Also, we convert UTF-8 to ASCII ignoring all bad characters sent by users
        print '@%s: %s\n' % (decoded['user']['screen_name'], decoded['text'].encode('ascii', 'ignore'))
        return True

    def on_error(self, status):
        print status

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

stream = tweepy.Stream(auth, TweetListener())

t = threading.Thread(target=stream.filter, kwargs={'track':['ISDC', 'PySD', 'ISDC15', 'Trump']})
t.daemon = True
t.start()

#make the animation
def animate(t):
    global counter
    #run the simulation forward
    time = model.components.t+dt
    model.run({'tweeting':counter}, 
              return_timestamps=time,
              return_columns=['tweeting', 'posts_on_timeline'],
              initial_condition='current',
              collect=True)
    out = model.get_record()
    ax.plot(out['tweeting'], 'r', label='Tweeting')
    ax.plot(out['posts_on_timeline'], 'b', label='Posts on Timeline')
    counter = 0

#set the animation parameters
fps=1
seconds=60*30
dt=1./fps    

#set up the plot
fig, ax = plt.subplots()
ax.set_xlim(0,seconds)
title = ax.set_title('Expected Twitter Messages on First Page of Feed')
ax.set_xlabel('Seconds')
ax.set_ylabel('Posts, Posts/second')
    
#reset the counter to start fresh.
counter=0    
    
# call the animator.
animation.FuncAnimation(fig, animate, repeat=False,
                        frames=seconds*fps, interval=1000./fps, 
                        blit=False)



