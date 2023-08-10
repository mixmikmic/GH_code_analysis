import praw

#this is a read-only instance
reddit = praw.Reddit(user_agent='first_scrape (by /u/dswald)',
                     client_id='TyAK1zSuAvQjmA', 
                     client_secret="uxHGsL0zNODbowN6umVnBWpqLAQ")


subreddit = reddit.subreddit('Portland')
hot_python = subreddit.hot(limit = 3)

for submission in hot_python:
    if not submission.stickied:
        print('Title: {}, ups: {}, downs: {}, Have we visited: {}'.format(submission.title,
                                                                          submission.ups,
                                                                          submission.downs,
                                                                          submission.visited))
        comments = submission.comments #unstructured
        for comment in comments:
            print (20*'-')
            print ('Parent ID:', comment.parent())
            print ('Comment ID:', comment.id)
            print (comment.body)
            
            if len(comment.replies) > 0:
                for reply in comment.replies:
                    print ('Reply:', reply.body) 
                    
                #option to continue digging into reply, but this is computationally heavy
                    

subreddit = reddit.subreddit('Portland')
hot_python = subreddit.hot(limit = 3) #need to view >2 to get past promoted posts

for submission in hot_python:
    if not submission.stickied: #top 2 are promoted posts, labeled as 'stickied'
        print('Title: {}, ups: {}, downs: {}, Have we visited: {}'.format(submission.title,
                                                                          submission.ups,
                                                                          submission.downs,
                                                                          submission.visited))
        comments = submission.comments.list() #unstructured
        for comment in comments:
            print (20*'-')
            print ('Parent ID:', comment.parent())
            print ('Comment ID:', comment.id)
            print (comment.body)

subreddit = reddit.subreddit('Portland')
hot_post = subreddit.hot(limit = 3)



for submission in hot_post:
    if not submission.stickied:
        print('Title: {}, ups: {}, downs: {}, Have we visited: {}'.format(submission.title,
                                                                          submission.ups,
                                                                          submission.downs,
                                                                          submission.visited))
        submission.comments.replace_more(limit=0) #this needs to be strung in here to view longer threads
        #this will throttle for bigger pages...
        for comment in submission.comments.list():
            print (20*'-')
            print ('Parent ID:', comment.parent()) #context builder
            print ('Comment ID:', comment.id) #unique id
            print (comment.body)

subreddit = reddit.subreddit('Portland')
hot_post = subreddit.hot(limit = 3)

submission = reddit.submission(url = 
                               'https://www.reddit.com/r/Portland/comments/75bi8h/psa_it_is_legal_for_bicyclist_to_ride_down/')
#submission.comment_sort = 'top' #I'm not sure this really works...


print('Title: {}, ups: {}, downs: {}, Have we visited: {}'.format(submission.title,
                                                                  submission.ups,
                                                                  submission.downs,
                                                                  submission.visited))
submission.comments.replace_more(limit=0)
for comment in submission.comments.list():
    print (20*'-')
    print ('Parent ID:', comment.parent()) #context builder
    print ('Comment ID:', comment.id) #unique id
    print (comment.body)

subreddit = reddit.subreddit('Portland')

for comment in subreddit.stream.comments():
    try:
        parent_id = str(comment.parent())
        original = reddit.comment(parent_id)
        print('parent:')
        print(original.body)
        
        print('reply:')
        print(comment.body)

    except praw.exceptions.PRAWException as e:
        pass

    



