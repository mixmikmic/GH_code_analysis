import urllib.request
import json
from bs4 import BeautifulSoup

def get(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    response = urllib.request.urlopen(req).read().decode('utf-8')
    return response

# return all the forum names of dcard
def get_forums():
    url = "https://www.dcard.tw/_api/forums"
    forums = json.loads(get(url))
    return [f['alias'] for f in forums]

# return the first 30 posts in the forum
def get_forum_posts(forum_name, before=None):
    url = "https://www.dcard.tw/_api/forums/{}/posts?popular=false".format(forum_name)
    if before:
        url += "&before={}".format(before)
    return json.loads(get(url))

def get_single_post(pid):
    url = "https://www.dcard.tw/_api/posts/{}".format(pid)
    return json.loads(get(url))

def get_comment_by_pid(pid):
    url = "https://www.dcard.tw/_api/posts/{}/comments".format(pid)
    return json.loads(get(url))

print(get_forums())

print(get_forum_posts("movie"))

print(get_single_post(227608258))

print(get_comment_by_pid(227608258))

