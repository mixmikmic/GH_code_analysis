import requests

def get_story(story_id):
    url = 'https://hacker-news.firebaseio.com/v0/item/%d.json' % story_id
    resp = requests.get(url)
    return resp.json()

def get_top_stories():
    url = 'https://hacker-news.firebaseio.com/v0/topstories.json'
    resp = requests.get(url)
    all_stories = [get_story(sid) for sid in resp.json()[:10]]
    return all_stories

import urllib3.contrib.pyopenssl
urllib3.contrib.pyopenssl.inject_into_urllib3()

top_stories = get_top_stories()

top_stories[:5]

MATCHING = (
    ('Python', '(p|P)ython'),
    ('Ruby', '(r|R)uby'),
    ('JavaScript', 'js|(J|j)ava(s|S)cript'),
    ('NodeJS', 'node(\.?)(?:\js|JS)'),
    ('Java', '(j|J)ava[^(S|s)cript]'),
    ('Objective-C', 'Obj(ective?)(?:\ |-)(C|c)'),
    ('Go', '(g|G)o'),
    ('C++',  '(c|C)(\+)+')
)

import re

def count_languages():
    stories = get_top_stories()
    final_tallies = {}
    for s in stories:
        long_string = u'{} {}'.format(s.get('title'), s.get('url'))
        for language, regex in dict(MATCHING).items():
            if re.search(regex, long_string):
                if language not in final_tallies.keys():
                    final_tallies[language] = {
                        'score': s.get('score'),
                        'descendants': s.get('descendants')}
                else:
                    final_tallies[language]['score'] += s.get('score')
                    final_tallies[language][
                        'descendants'] += s.get('descendants')
    return final_tallies

count_languages()

get_ipython().magic('load solutions/regex_solution.py')



