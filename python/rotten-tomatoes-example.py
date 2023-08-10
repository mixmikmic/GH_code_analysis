get_ipython().run_cell_magic('javascript', '', 'IPython.OutputArea.auto_scroll_threshold = 10;')

from processors import *
# We'll be using the server in several examples
# NOTE: you can stop it manually with API.stop_server()
API = ProcessorsAPI(port=8886, keep_alive=True)

# CoreNLP's sentiment scores range from (1 (very negative) to 5 (very positive)`)
API.sentiment.corenlp.score_text("I'm so happy!")

from snappytomato import *

# path to api key
api_key_file = "../rt.key"
api_key = load_api_key(api_key_file)
snappy = RT(api_key)

movie = snappy.movies.movie_by_title('big trouble in little china')

summary = """
title:\t\t\t"{}"
audience score:\t\t{}
critic consensus:\t{}
""".format(movie.title, movie.audience_score, movie.critics_consensus or "unknown")
print(summary)

reviews = movie.reviews
print("{} reviews for \"{}\"".format(len(movie.reviews), movie.title))
for r in reviews:
    summary = """
    critic: {}
    quote: {}
    publication: {}
    source: {}
    freshness: {}
    original score: {}""".format(r.critic, r.quote, r.publication, r.source, r.freshness, r.original_score if hasattr(r, "original_score") else "None")
    print(summary)

from bs4 import BeautifulSoup
import requests

# hmmm...not all the reviews are positive...
# I guess not everyone is enlightened.
# Well, let's look at a review we know
# to be "fresh"
r = reviews[-2]
print("Critic: {}".format(r.critic))
print("Publication: {}".format(r.publication))
print("Freshness: {}".format(r.freshness))
# not every review has a link
url = r.links.get("review", None)
article_text = ""
if url:
    response = requests.get(url)
    html = response.text
    soup = BeautifulSoup(html, "html.parser")
    article_text = "\n".join([p.text for p in soup.find_all("p")])

article_text

sentiment_scores = API.sentiment.corenlp.score_text(article_text)
print(sentiment_scores)

API.sentiment.corenlp.score_text("I'm so happy!")

API.sentiment.corenlp.score_text(":)")

