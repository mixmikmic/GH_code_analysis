import pandas as pd
import json
import numpy as np
import nltk
from nltk.corpus import stopwords
import re
import gensim
import math

global AVG_TAGS_PER_VIDEO, US_CA_GB_TOKEN_CORPUS, US_VIDEOS_DF, US_FINAL_DF
global CA_VIDEOS_DF, CA_FINAL_DF, GB_VIDEOS_DF, GB_FINAL_DF, US_CA_GB_FINAL_DF

#get rid of the punctuations and set all characters to lowercase
RE_PREPROCESS = r'\W+|\d+' #the regular expressions that matches all non-characters

#get rid of punctuation and make everything lowercase
#the code belows works by looping through the array of text
#for a given piece of text we invoke the `re.sub` command where we pass in the regular expression, a space ' ' to
#subsitute all the matching characters with
#we then invoke the `lower()` method on the output of the re.sub command
#to make all the remaining characters
#the cleaned document is then stored in a list
#once this list has been filed it is then stored in a numpy array

RE_REMOVE_URLS = r'http\S+'

def processFeatures(desc):
    try:
        desc = re.sub(RE_REMOVE_URLS, ' ', desc)
        return re.sub(RE_PREPROCESS, ' ', desc)
    except:
        return " "

def processDataFrame(data_frame, country_code='US'):
    data_frame.sort_values(by=['video_id', 'trending_date'], ascending=True, inplace=True)
    grouped_videos = data_frame.groupby(['video_id']).last().reset_index()
    
    #Reading categories from the json file depending on country_code
    json_location = './data/' + country_code +'_category_id.json'
    with open(json_location) as data_file:
        data = json.load(data_file)    
    categories = []
    for item in data['items']:
        category = {}
        category['category_id'] = int(item['id'])
        category['title'] = item['snippet']['title']
        categories.append(category)

    categories_df = pd.DataFrame(categories)
    # Merging videos data with category data
    final_df = grouped_videos.merge(categories_df, on = ['category_id'])
    final_df.rename(columns={'title_y': 'category', 'title_x': 'video_name'}, inplace=True)
    
    # Creating a features column that consists all features used for prediction.
    # Also creating a corpus column that consists of all data required to train the model.
    final_df['video_features'] = ''
    final_df['video_corpus'] = ''
    
    if final_df['video_name'].astype(str) is not None:
        final_df['video_features'] += final_df['video_name'].astype(str)

    if final_df['channel_title'].astype(str) is not None:
        final_df['video_features'] += final_df['channel_title'].astype(str)
        
    if final_df['description'].astype(str) is not None:
        final_df['video_features'] += final_df['description'].astype(str)
    
    final_df['video_corpus'] += final_df['video_features']
    if final_df['tags'].astype(str) is not None:
        final_df['video_corpus'] += final_df['tags'].astype(str)
    
        
    final_df['video_features'] = final_df['video_features'].apply(processFeatures)
    final_df['video_corpus'] = final_df['video_corpus'].apply(processFeatures)
    return final_df

def removeNonEngAndStopwords(documents):
    stopwords_list = stopwords.words('english')
    processed_corpus = []
    for document in documents:
        processed_document = []
        for word in document.split():
            try:
                if word not in stopwords_list and word.encode(encoding='utf-8').decode('ascii'):
                    processed_document.append(word)
            except UnicodeDecodeError:
                # Can log something here
                pass
        processed_corpus.append(processed_document)
    return processed_corpus

def processCorpus(feature_corpus):
    feature_corpus = [comment.lower() for comment in feature_corpus]
    processed_feature_corpus = removeNonEngAndStopwords(feature_corpus)
    return processed_feature_corpus

def trainModel(token_corpus, model_name = 'word2vec_model.w2v'):
    model = gensim.models.Word2Vec(sentences=token_corpus, min_count=1, size = 32)
    model.train(token_corpus, total_examples=model.corpus_count, epochs=model.iter)
    model.save(model_name)
    return model

def recommendTags(word2vec_model, input_words = ['trump', 'president'], number_of_tags = 10, model_name = 'word2vec_model.w2v'):
    global US_CA_GB_TOKEN_CORPUS
    tags = []
         
    try:
        word2vec_model = gensim.models.Word2Vec.load(model_name)
        tags = word2vec_model.most_similar(positive=input_words, topn=number_of_tags)
    except FileNotFoundError:
        word2vec_model = trainModel(US_CA_GB_TOKEN_CORPUS, model_name)
        try:
            tags = word2vec_model.most_similar(positive=input_words, topn=number_of_tags)
        except:
            US_CA_GB_TOKEN_CORPUS.append(input_words)
            word2vec_model.build_vocab(US_CA_GB_TOKEN_CORPUS, update=True)
            word2vec_model.train(US_CA_GB_TOKEN_CORPUS, total_examples=word2vec_model.corpus_count, epochs=word2vec_model.iter)
            word2vec_model.save(model_name)
            tags = word2vec_model.most_similar(positive=input_words, topn=number_of_tags)
    except:
        US_CA_GB_TOKEN_CORPUS.append(input_words)
        word2vec_model.build_vocab(US_CA_GB_TOKEN_CORPUS, update=True)
        word2vec_model.train(US_CA_GB_TOKEN_CORPUS, total_examples=word2vec_model.corpus_count, epochs=word2vec_model.iter)
        word2vec_model.save(model_name)
        tags = word2vec_model.most_similar(positive=input_words, topn=number_of_tags)
    
    return tags

def calculateAvgTagsPerVideo():
    total_tags = 0
    for tag_list in US_CA_GB_FINAL_DF['tags'].values:
        total_tags += len(tag_list.split('|'))
    return math.ceil(total_tags/len(US_CA_GB_FINAL_DF))

def initializeAndFetchRecommendations(video_name = None, channel_title = None, video_category = None, description = None):
    global US_VIDEOS_DF, US_FINAL_DF, CA_VIDEOS_DF, CA_FINAL_DF, GB_VIDEOS_DF, GB_FINAL_DF
    global US_CA_GB_FINAL_DF, US_CA_GB_FINAL_DF, AVG_TAGS_PER_VIDEO, US_CA_GB_TOKEN_CORPUS
    US_VIDEOS_DF = pd.read_csv('./data/USvideos.csv')
    US_FINAL_DF = processDataFrame(US_VIDEOS_DF, country_code='US')
    
    CA_VIDEOS_DF = pd.read_csv('./data/CAvideos.csv')
    CA_FINAL_DF = processDataFrame(CA_VIDEOS_DF, country_code='CA')
    
    GB_VIDEOS_DF = pd.read_csv('./data/GBvideos.csv')
    GB_FINAL_DF = processDataFrame(GB_VIDEOS_DF, country_code='GB')
        
    US_CA_GB_FINAL_DF = pd.concat([US_FINAL_DF, CA_FINAL_DF, GB_FINAL_DF])
    US_CA_GB_FINAL_DF.reset_index(inplace=True)
    
    US_CA_GB_TOKEN_CORPUS = processCorpus(US_CA_GB_FINAL_DF['video_corpus'].values)
    US_CA_GB_FINAL_DF['video_features'] = processCorpus(US_CA_GB_FINAL_DF['video_features'].values)
    US_CA_GB_FINAL_DF['video_corpus'] = US_CA_GB_TOKEN_CORPUS
        
    AVG_TAGS_PER_VIDEO = calculateAvgTagsPerVideo()
    word2vec_model = None
    
    input_list = []
    if (video_name is not None or channel_title is not None or
        video_category is not None or description is not None):
        frontEndInput = frontEndInput = video_name + ' ' + channel_title + ' ' +  video_category + ' ' + description + ' '
        for word in frontEndInput.split(' '):
            if word not in stopwords.words('english') and len(word.strip()) > 0:
                input_list.append(word.lower())

    if input_list != []:
        return recommendTags(word2vec_model, input_words=input_list,
                         number_of_tags=AVG_TAGS_PER_VIDEO,
                         model_name = 'word2vec_model.w2v')

    return recommendTags(word2vec_model, input_words=['trump', 'president'],
                         number_of_tags=AVG_TAGS_PER_VIDEO,
                         model_name = 'word2vec_model.w2v')
        
    

recommendations = initializeAndFetchRecommendations()

recommendations

the_file = open("recommendations.txt","w+")
for recommendation in recommendations:
    the_file.write(recommendation[0] + ' ')
the_file.close()

initializeAndFetchRecommendations(video_name = 'What is data science',
                                  channel_title = 'CNN', 
                                  video_category = 'Education', 
                                  description = 'data science related')

np.random.seed(seed=13579)
us_ca_gb_final_df_shuffled = US_CA_GB_FINAL_DF.iloc[np.random.permutation(len(US_CA_GB_FINAL_DF))]

train_size = 0.80
us_ca_gb_df_train = us_ca_gb_final_df_shuffled[:int((train_size)*len(us_ca_gb_final_df_shuffled))]
us_ca_gb_df_test = us_ca_gb_final_df_shuffled[int((train_size)*len(us_ca_gb_final_df_shuffled)):]

try:
    w2v_train_model = gensim.models.Word2Vec.load('w2v_train_model.w2v')
except FileNotFoundError:
    w2v_train_model = gensim.models.Word2Vec(sentences=us_ca_gb_df_train['video_corpus'], min_count=1, size = 32)
    w2v_train_model.train(us_ca_gb_df_train['video_corpus'].values, total_examples=w2v_train_model.corpus_count, epochs=w2v_train_model.iter)
    w2v_train_model.save('w2v_train_model.w2v')    

us_ca_gb_df_test = us_ca_gb_df_test[us_ca_gb_df_test['video_features'].map(len) > 0]

predicted_tags = []
for idx in us_ca_gb_df_test.index:
    video_features = us_ca_gb_df_test.loc[idx, 'video_features']
    tag_probability_list = recommendTags(w2v_train_model, input_words=video_features, 
                                         number_of_tags=AVG_TAGS_PER_VIDEO, 
                                         model_name = 'w2v_train_model.w2v')
    predicted_tags.append([tag[0] for tag in tag_probability_list if len(tag_probability_list) != 0])

        
        

    

us_ca_gb_df_test['predicted_tags'] = predicted_tags

us_ca_gb_df_test['tags'] = us_ca_gb_df_test['tags'].apply(processFeatures)

match_found = 0
count = 0
for idx in us_ca_gb_df_test.index:
    tag_list = us_ca_gb_df_test.loc[idx,'tags'].lower()
    tag_list = tag_list.split(' ')
    predicted_tag_list = us_ca_gb_df_test.loc[idx, 'predicted_tags']
    
    for i in range(len(tag_list)):
        if tag_list[i] in predicted_tag_list:
            match_found += 1
            break
    count += 1
print('Match found: ', match_found )
print('Accuracy: ', match_found/len(us_ca_gb_df_test))
    
    

def computeSimilarity(word1, word2):
    try:
        return w2v_train_model.wv.similarity(word1, word2)
    except:
        return 0
    

for idx in us_ca_gb_df_test.index:
    tag_list = us_ca_gb_df_test.loc[idx,'tags'].lower()
    tag_list = tag_list.split(' ')
    predicted_tag_list = us_ca_gb_df_test.loc[idx, 'predicted_tags']
    avg_similarity_per_row = 0
    avg_similarity_scores = []
    
    for predicted_tag in predicted_tag_list:
        similarity_score = -2
        for tag in tag_list:
            similarity_score = max(similarity_score, computeSimilarity(predicted_tag, tag))
        avg_similarity_per_row += similarity_score
    
    avg_similarity_scores.append(avg_similarity_per_row / len(tag_list))

cosine_similarity_value =sum(avg_similarity_scores)/len(avg_similarity_scores)
print('Similarity Value: ', cosine_similarity_value)
        
            

the_file = open("corpus_of_strings.txt","w+")

input = ''
for token_list in US_CA_GB_TOKEN_CORPUS:
    for token in token_list:
        input += token + ' '

the_file.write(input)

the_file.close()



