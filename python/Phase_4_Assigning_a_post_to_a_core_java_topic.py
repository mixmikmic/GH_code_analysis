import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stopWords = set(stopwords.words('english'))

new_topic = True
java_topic_dict = {}
java_topic = ''
tmp_list = []

with open('./data/java_topics.txt') as file:
    for line in file:   
        #create a new topic
        if line in '\n':
            java_topic_dict[java_topic] = set(tmp_list)
            tmp_list = []
            new_topic = True
            continue
            
        if new_topic:
            #assign the current line to be the java topic
            java_topic = ' '.join(line.split()[1:])
            new_topic = False
            continue
        
        #split each line into individual words and append
        for el in line.split()[1:]:
            #stopword removal
            if el not in stopWords and len(el) > 1:
                tmp_list.append(el.lower().replace('-', ''))
        

java_topic_dict.keys()

java_topic_keys = list(java_topic_dict.keys())

# stores the refined topic terms
new_java_topic_dict = {}

#scan through each topic
for key in java_topic_keys:
    
    #tmp list to hold all the terms related to topic
    tmp_list = []
    for item in java_topic_keys:
        #we do not want to add the terms related to the current topic to the main list
        #when detected move on to next
        if item == key:
            continue
        else:
            #for each term related to the topic append them to the tmp list
            for el in list(java_topic_dict[item]):
                tmp_list.append(el)
    #create a set form the tmp list containing terms from all the topics except the current topic (key)
    #this creates our stop list
    stop_list = set(tmp_list)
    
    #initialise a filtered_list to store unique values for each topic
    filtered_list = []
    for el in list(java_topic_dict[key]):
        #if it doesnt appear in our stop list dont append it
        if el not in stop_list:
            filtered_list.append(el)

    #add the key to the list
    filtered_list.append(key.lower())
    #creates a new list of unique terms related to that topic. 
    new_java_topic_dict[key] = filtered_list

import pandas as pd
df = pd.read_csv('./data/filtered_cleaned_posts_no_frameworks_no_alt_lang.csv')

df.shape

# word2vec function converts the word into a vector
def word2vec(word):
    from collections import Counter
    from math import sqrt

    # count the characters in word
    cw = Counter(word)
    # precomputes a set of the different characters
    sw = set(cw)
    # precomputes the "length" of the word vector
    lw = sqrt(sum(c*c for c in cw.values()))

    # return a tuple
    return cw, sw, lw

# calculates the cosine similarity between 2 vectors i.e. words
def cosdis(v1, v2):
    # which characters are common to the two words?
    common = v1[1].intersection(v2[1])
    # by definition of cosine distance we have
    return sum(v1[0][ch]*v2[0][ch] for ch in common)/v1[2]/v2[2]

'''
    Description:
        takes an argument of a list of tags and checks two things:
        
        1)  checks to see if any of the tags are related directly to a core topic e.g. generics, classes, exceptions
            if the score is greater than 95% then it returns the key that the tag scored greater than 95% against.
        
        2)  If there is no match over 95% for the keys, the tags are then compared to each of the terms in all 
            of the topics, each topics terms are loaded in and the cosine similarity is checked against each word
            the highest scoring word is stored. For example if there are three tags, each tag will be compared to the 
            rest of the terms in a given topic. The highest score for each tag is appended to a total score, 
            this generates a percentage of how likely it is related to that topic. The highest scoring topic is returned
    
    args: tag_list
        list of tags related to a given post
        
'''
def assign_java_topic(tag_list):
    
    #first check if it contains a core topic tag
    for tag in tag_list:
        tag = tag.replace('-', '')
        va = word2vec(tag)
        for jkey in java_topic_keys:
            vb = word2vec(jkey.lower())
            #if the cosine similarity is greater than 95% return that key as the category
            if cosdis(va, vb) > 0.95:
                return jkey, cosdis(va,vb)

    #stores the most likely topic
    max_topic_name = ''
    #holds the max topic score for the set of tags
    max_topic_score = 0

    for key in java_topic_keys:
        #stores the score of the current topic
        topic_score = 0
        #each tag is converted to a vector
        for tag in tag_list:
            tag = tag.replace('-', '')
            va = word2vec(tag)
            max_tag_score = 0

            #for every element in the current topic e.g. classes
            for el in new_java_topic_dict[key]:
                #vb is the current term in the current topic converted to vector format
                vb = word2vec(el)
                #calculate the cosine similarity
                score = cosdis(va,vb)

                #if the score is greater than the current max tag score then 
                # assign this as the new max
                if score > max_tag_score:
                    max_tag_score = score
            #increment the overall topic score using the max tag score
            topic_score += max_tag_score
            
        #check if this is the highest scoring topic score
        if topic_score > max_topic_score:
            max_topic_score = topic_score
            max_topic_name = key
    
    #returns the highest scoring topic along with the score as a percentage. 
    return max_topic_name, max_topic_score/len(tag_list)

test = df.head(10)

for index, row in test.iterrows():   
    topic_name, topic_score = assign_java_topic(test.loc[index, 'Tags'].split())
    print(test.loc[index, 'Tags'].split())
    print(topic_name)
    print(topic_score)
    print('\n')

def extract_title_keywords(title):
    from textblob import TextBlob
    s_list = []
    blob = TextBlob(title)
    for word, pos in blob.tags:
        if pos == 'NN' or pos == 'JJ' or pos == 'VB':
            s_list.append(word)
    return s_list

s_test = ["How can I make this java generic cast?", "How to convert nanoseconds to seconds using the TimeUnit enum?",
         "How can I sort the keys of a Map in Java?", "How does Java convert int into byte?", 
          "how to convert byte array to string and vice versa ", "How do I make a Class extend Observable when it has extended another class too?",
         "How to check if an IP address is the local host on a multi-homed system?"]
for s in s_test: 
    res = extract_title_keywords(s)
    topic_name, topic_score = assign_java_topic(res)
    print(res)
    print(topic_name)
    print(topic_score)
    print('\n')

#store the codes
java_topic_codes = {}
count = 0

for key in java_topic_keys:
    java_topic_codes[key] = count
    count += 1
#sets a default key
java_topic_codes['other'] = count    

java_topic_codes

# check if tags column empty
df.Tags[df.Tags == "  -7"].count()

#initialise the topic column
df["Topic"] = None
for index, row in df.iterrows():
    #if tags are not present use the title from the post
    if df.loc[index, "Tags"] == "  " or df.loc[index, "Tags"] in "-8" or df.loc[index, "Tags"] in "-7":
        #extract the keywords from the title
        res = extract_title_keywords(df.loc[index, "Title"])
        if len(res) > 0:
            topic_name, topic_score = assign_java_topic(res)
            df.loc[index, "Topic"] = java_topic_codes[topic_name]
    #otherwise use the tags
    else:
        topic_name, topic_score = assign_java_topic(df.loc[index, "Tags"].split())
        
        if topic_name == '':
            topic_name = "other"
        df.loc[index, "Topic"] = java_topic_codes[topic_name]

df[df.Topic == 4]

df.to_csv("./data/java_questions_including_topics.csv", index=False)

for key in java_topic_keys:
    print(key)



