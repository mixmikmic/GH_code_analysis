import re
from TwitterSearch import *

def strip_whitespace(s):
    return re.sub(r'\s\s+', ' ', s)

try:
    tso = TwitterSearchOrder()
    tso.arguments.update({'tweet_mode':'extended'})
    tso.set_include_entities(False)
    tso.set_keywords(['το', '-filter:retweets', '-filter:replies'])
    tso.set_language('el')
    tso.set_locale('cy')
#     tso.set_result_type('popular')

    ts = TwitterSearch(
        consumer_key = 'SCbnEI8itr0p9e7bYiMZGkHaI',
        consumer_secret = 'xrWiU7Fwmv2hUwth3vbtXEOyjuEAlI2WGbwtkYEtb5IpzBSA4B',
        access_token = '1032285068-9ImnboGg1EHthmKSEgCh4F6k3csuyB5GWGDg355',
        access_token_secret = 'e3knnkUvZqkepPDDPwTNXxw39ZhpYdWdguoEqCBQ3UsxO'
     )
    
    save_file = open('./Data/smg_twitter.txt', 'a', encoding='utf-8')
    
    for tweet in ts.search_tweets_iterable(tso):
        print(tweet['full_text'], '\n')
        save_file.write(strip_whitespace(tweet['full_text']))
        save_file.write('\n')

    save_file.close()
        
except BaseException as e:
    print(e)
    
finally:
    if not save_file.closed:
        save_file.close()

try:
    tuo = TwitterUserOrder('NeinQuarterly')
                           
    ts = TwitterSearch(
        consumer_key = 'SCbnEI8itr0p9e7bYiMZGkHaI',
        consumer_secret = 'xrWiU7Fwmv2hUwth3vbtXEOyjuEAlI2WGbwtkYEtb5IpzBSA4B',
        access_token = '1032285068-9ImnboGg1EHthmKSEgCh4F6k3csuyB5GWGDg355',
        access_token_secret = 'e3knnkUvZqkepPDDPwTNXxw39ZhpYdWdguoEqCBQ3UsxO'
     )
    
    save_file = open('./Data/cg_twitter.txt', 'a', encoding='utf-8')
    
    for tweet in ts.search_tweets_iterable(tuo):
        save_file.write(strip_whitespace(tweet['full_text']))
        save_file.write('\n')
        
    save_file.close()
    
except TwitterSearchException as e:
    print(e)

