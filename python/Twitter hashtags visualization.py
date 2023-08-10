import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

from extractor import TweetsExtractor

extractor = TweetsExtractor()
data = extractor.extract("pybites")
data.head()

from extractor import TweetsExtractor
from analyzer import TweetsAnalyzer


extractor = TweetsExtractor()
analyzer = TweetsAnalyzer(extractor)
analyzer.analyze("pybites")

# We use dict() to convert it to a normal dict:
dict(analyzer.hashtags)

fav, rt = analyzer.trending_tweets()

fav_tw = analyzer.data['Tweets'][fav]
rt_tw = analyzer.data['Tweets'][rt]

# Max FAVs:
print("The tweet with more likes is: \n{}".format(fav_tw))
print("Number of likes: {}".format(analyzer.data['Likes'][fav]))
print("{} characters.\n".format(analyzer.data['len'][fav]))

# Max RTs:
print("The tweet with more retweets is: \n{}".format(rt_tw))
print("Number of retweets: {}".format(analyzer.data['RTs'][rt]))
print("{} characters.\n".format(analyzer.data['len'][rt]))

top = analyzer.top_hashtags(top=10)
for i, hashtag in enumerate(top):
    print("{}. {}: {}".format(i+1, hashtag[0], hashtag[1]))

from visualizer import TweetsVisualizer

visualizer = TweetsVisualizer(analyzer)
visualizer.lengths()

visualizer.likes()

visualizer.retweets()

visualizer.retweets()
visualizer.likes()
plt.title("Retweets vs Likes");

mask = visualizer.create_mask("../imgs/twird.png", threshold=100)
mask = 255 - mask
plt.imshow(mask, cmap="gray")
plt.grid('off')

visualizer.wordcloud(mask)
plt.savefig("../imgs/wordcloud_twird.png", dpi=300)

mask = visualizer.create_mask("../imgs/pybites.png", threshold=230)
plt.imshow(mask, cmap="gray")
plt.grid('off')

visualizer.wordcloud(mask)
plt.savefig("../imgs/wordcloud_pybites.png", dpi=300)

mask = visualizer.create_mask("../imgs/turing.jpg", threshold=150)
plt.imshow(mask, cmap="gray")
plt.grid('off')

visualizer.wordcloud(mask)
plt.savefig("../imgs/wordcloud_turing.png", dpi=300)



