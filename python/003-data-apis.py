























import json
import requests

# get_article_revisions is a function that takes an article title in
# wikipedia and return a list of all the revisions and meatadata for
# that article
def get_article_revisions(title):
    revisions = []

    # create a base url for the api and then a normal url which is initially just a copy of it
    wp_api_base = "http://en.wikipedia.org/w/api.php/?action=query&titles=%(article_title)s&prop=revisions&rvprop=flags|timestamp|user|size|ids&rvlimit=500&format=json"
    wp_api_base = wp_api_base % {'article_title': title }
    wp_api_url = wp_api_base

    # we'll repeat this forever (i.e., we'll only stop when we find the "break" command)
    while True:
        # the first line open the urls but also handles unicode urls
        call = requests.get(wp_api_url)
        api_answer = call.json()

        # get the list of pages from the json object
        pages = api_answer["query"]["pages"]

        # for every pages (there should always be only one) get the revisions
        for page in pages.keys():
            if "revisions" in pages[page].keys():
                query_revisions = pages[page]["revisions"]

                # for every revision, we do first do cleaning up
                for rev in query_revisions:
                    # lets continue/skip if the user is hidden
                    if "userhidden" in rev.keys():
                        continue

                    # 1: add a title field for the article because we're going to mix them together
                    rev["title"] = title

                    # 2: lets "recode" anon so it's true or false instead of present/missing
                    if "anon" in rev.keys():
                        rev["anon"] = True
                    else:
                        rev["anon"] = False

                    # 3: letst recode "minor" in the same way
                    if "minor" in rev.keys():
                        rev["minor"] = True
                    else:
                        rev["minor"] = False

                    # we're going to change the timestamp to make it work a little better in excel and similar
                    rev["timestamp"] = rev["timestamp"].replace("T", " ")
                    rev["timestamp"] = rev["timestamp"].replace("Z", "")

                    # finally save the revisions we've seen to a varaible
                    revisions.append(rev)

        # if there is a query-continue, it means there are more
        if 'query-continue' in api_answer.keys():
            # we will grab the rvcontinue token, insert it, and head back to the start of the loop
            rvcontinue = api_answer["query-continue"]["revisions"]["rvcontinue"]
            wp_api_url = wp_api_base + "&rvcontinue=%(continue_from)s" % {'continue_from' : rvcontinue}
        else:
            # no continue means we're done
            break

    # return all the revisions for this page
    return(revisions)

category = "Avengers_(comics)"

# we'll use another api called catscan2 to grab a list of pages in
# categories and subcategories. it works like the other apis we've
# studied!
url_catscan = 'http://tools.wmflabs.org/catscan2/catscan2.php?depth=10&categories=%(category)s&doit=1&format=json'
url_catscan = url_catscan % {'category' : category}
call = requests.get(url_catscan)
articles = json.loads(call.content)
articles = articles["*"][0]["a"]["*"]

# open a filie to write all the output
# output = codecs.open("hp_wiki.csv", "wb", "utf-8")
output = open('avengers_wiki.csv', 'w')

output.write(",".join(["title", "user", "timestamp", "size", "anon", "minor", "revid"]) + "\n")

# for every article
for article in articles:

    # first grab tht title
    title = article["title"]

    # get the list of revisions from our function and then interating through it printinig it out
    revisions = get_article_revisions(title)
    for rev in revisions:
        output.write(",".join(['"' + rev["title"] + '"', '"' + rev["user"] + '"',
                               rev["timestamp"], str(rev["size"]), str(rev["anon"]),
                               str(rev["minor"]), str(rev["revid"])]) + "\n")

# close the file, we're done here!
output.close()
    
    



