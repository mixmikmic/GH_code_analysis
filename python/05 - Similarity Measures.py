import pandas as pd
import pickle
import numpy as np
import pandas as pd
import pickle
import numpy as np

sys.path.append('../vectorsearch/')
import nltk_helper
import gensim

# Load the bar review dataset 
review = pd.read_pickle('../output/bar_reviews_cleaned_and_tokenized.pickle')
review.head(2)

model = gensim.models.Doc2Vec.load('../output/doc2vec_bars.model')



get_ipython().magic('autoreload 2')




def get_mean_word_vector(words, model):
    '''
    Given a dict of words and weights, determine the mean vector.
    words : dict
        Dict containing search terms and weights
    ''' 
    mean = np.zeros(model[words.keys()[0]].shape[0])
    vec = np.sum([model[word]*weight for (word, weight) in words.items()])
    return vec/np.dot(vec,vec)
    
def get_mean_doc_vector(review_ids, model):
    '''
    Given a list of review_ids, determine the mean vector.
    
    reviews : pandas_dataframe
    
    words : dict
        Dict containing search terms and weights
    ''' 
    # Indices of documents 
    idx = np.array(model.docvecs.indexed_doctags(review_ids)[0])
    return np.sum(model.docvecs[idx], axis=0)
    
    
def dot_business(words, business_id, review):
    '''
    Find the mean cosine similarity over all reviews given a dict of words, and a business_id     
    
    words : dict
        Dict containing search terms and weights
    business_id : str
        str containing the business_id of interest.
        
    Returns
    ---------------
    similarity: float
        average cosine similarity of the word vector dotted onto the business reviews.  
    
    ''' 
    # Get the aggregate weighted word vector
    mean_words = get_mean_word_vector(words, model)
    # List the review ids associated with the given business
    review_ids = review_helper.get_review_ids(business_id, review)
    
    # Find similarity of the word vector and each review vector
    # First get the review indices in the doc2vec model 
    idx = np.array(model.docvecs.indexed_doctags(review_ids)[0])
    # Dot the business onto the mean words and average.
    # TODO: IS averaging the best?  Maybe we should take average over some threshold... 
    try:
        return np.max(np.dot(model.docvecs.doctag_syn0[idx], mean_words))
    except:
        print idx
business_ids = list(set(review.business_id.values[:10000]))
#print business_ids
cos_sim = [dot_business(words={'beer':10}, business_id=bus_id, review=review) for bus_id in business_ids]
print cos_sim
print np.argmax(cos_sim)
print review[business_ids[np.argmax(cos_sim)]==review.business_id].text.values

[len(review.review_id[review.business_id==id]) for id in set(review.business_id.values[:100])]

for word, sim in model.most_similar(positive=['beer', 'pricy']):
    print word, sim 


review.columns

businesses = pd.read_pickle('../input/yelp_academic_dataset_business.pickle')

bus_id = pd.read_pickle('../output/bar_ids.pickle')
businesses = businesses[businesses.business_id.isin(bus_id)]


def GetBusinessIds(city='Las Vegas'):
    return businesses.business_id[businesses['city']==city]
print len(GetBusinessIds())

lv_bus_ids = GetBusinessIds()
lv_businesses = businesses[businesses.business_id.isin(lv_bus_ids)]
#print lv_businesses.name
lv_reviews = review[review.business_id.isin(lv_bus_ids)]

bus_id_of_interest = lv_businesses.business_id[lv_businesses.name=="Fado Irish Pub"] # Business of interest... 
rev_of_interest = lv_reviews[lv_reviews.business_id.isin(bus_id_of_interest)]
rev_ids_of_interest = rev_of_interest.review_id.values

mean_bus_rev_vector = get_mean_doc_vector(rev_ids_of_interest, model)

model.most_similar(positive=[mean_bus_rev_vector], topn=20)

model.most_similar('snooker')



