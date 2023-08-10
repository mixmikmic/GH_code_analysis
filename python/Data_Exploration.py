get_ipython().system(' cd ../.. && mkdir CMV && cd CMV && curl -# -O https://chenhaot.com/data/cmv/cmv_20161111.jsonlist.gz && gunzip cmv_20161111.jsonlist.gz ')

import glob
import json
data = glob.glob('../../CMV/*')[0] # Get string for JSON file with data

f = open(data, 'rb')

for line in f:
    l = json.loads(line.decode('utf-8'))
    print(l)
    break

from IPython.display import Markdown

def show_post(cmv_post):
    md_format = "**{title}** \n\n {selftext}".format(**cmv_post)
    md_format = "\n".join(["> " + line for line in md_format.splitlines()])
    return Markdown(md_format)

show_post(l)

l.keys()

l['selftext']

l['title']

l['name']

l['author']

l['id']

l['created']

l['url']

l['comments'][2]

l['comments'][0].keys()

for c in l['comments']:
    try:
        print(c['author'])
        print(c['created'])
        print(c['body'])
        print('\n')
        print(c)
    except:
        pass

def has_delta(comment_text):
    if '∆' in comment_text:
        return True
    else:
        return False

print('∆')

for c in l['comments']:
    try:
        if has_delta(c['body']):
            print("DELTA DETECTED")
            print(c['body'])
    except:
        pass

f = open(data, 'rb')
posts_dict = {}
comments_dict = {}
for line in f:
    post = json.loads(line.decode('utf-8'))
    post_info = {
                 'title': post['title'],
                 'text': post['selftext'],
                 'author': post['author'],
                 'num_comments': post['num_comments'],
                 'time': post['created'],
                 'url': post['url'],
                 'name': post['name'],
                 'score': post['score']
    }
    comment_list = []
    for c in post['comments']:
        try:
            comment_list.append({
                'author' : c['author'],
                'time' : c['created'],
                'text': c['body'],
                'parent': c['parent_id'],
                'score': c['score'],
                'delta': has_delta(c['body'])

            })
        except: # Skip comments if they do not have these attributes
            pass 
    posts_dict[post['id']] = post_info
    comments_dict[post['id']]= comment_list
f.close()

pickle.dump(posts_dict, open('post_info.p','wb'))

pickle.dump(comments_dict, open('comment_info.p','wb'))

get_ipython().system('ls -lh')

