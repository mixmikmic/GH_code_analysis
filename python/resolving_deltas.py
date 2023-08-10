import pickle
posts = pickle.load(open('post_info.p','rb'))
comments = pickle.load(open('comment_info.p','rb'))

post_ids = list(posts.keys())

p = post_ids[1]

pst = posts[p]
cmt = comments[p]

pst

op = pst['author']
pname = pst['name']
print("OP: ", op, " Post name: ", pname, " Post ID: ", p)

for c in cmt:
    if c['delta']:
        print(c['author'])

for c in cmt:
    if c['delta']:
        print(c['author'])
        print(c['text'])
        print('\n')

for c in cmt:
    if c['delta']:
        parent = c['parent'].split('_')[1]
        for k in cmt:
            if k['id'] == parent:
                print(k['author'])

for c in cmt:
    if c['author'] != "DeltaBot":
        if c['delta']:
            parent = c['parent'].split('_')[1]
            for k in cmt:
                if k['id'] == parent:
                    print(k['author'])

def resolve_deltas(comments):
    """This function takes a set of comments in response to a post and 
    returns the usernames and comment IDs of the users who the OP 
    awarded a delta
    
    Returns
    
    list of tuples : Each tuple is an author name, comment ID pair."""
    #TODO: This could be made more efficient by turning comments into a dict rather
    # than a list of dicts.
    D = []
    for c in comments:
        if c['author'] != "DeltaBot": # Ignore deltabot
            if c['delta']:
                parent = c['parent'].split('_')[1]
                for k in comments: # loop through again to find parent
                    if k['id'] == parent:
                        D.append((k['author'],k['id']))
    return D

resolve_deltas(cmt)

deltas_dict = {}
for p in post_ids:
    # Get comments for that post
    deltas = resolve_deltas(comments[p])
    deltas_dict[p] = deltas

# Store deltas object in pickle
pickle.dump(deltas_dict, open('deltas_info.p','wb'))

deltas_dict



