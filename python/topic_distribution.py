topics = []
with open('./data/community.txt') as commF:
    for line in commF.readlines():
        topic = line.strip('\n').split()
        topics.append(topic)

topics

import json
threads = {}
with open('./data/2017-01thread.json') as f:
    data = json.load(f)
    for key in data:
        subred = data[key]
        for thread_id in subred:
            threads[thread_id] = subred[thread_id]
            
        

len(threads)

word_tf = {}
word_df = {}
for thread_id, thread in threads.items():
    lst = thread.split()
    word_tf[thread_id] = {}
    for topic in topics:
        for word in topic:
            if word in lst:
                tf = lst.count(word) 
                word_tf[thread_id][word] = tf
                word_df[word] = word_df.get(word, 0) + 1
    

word_tf

word_df

import math

topic_tfidf = {}
for thread_id in word_tf:
    topic_tfidf[thread_id] = []
    for topic in topics:
#         print(topic)
        tfidf = 0
        num_word = 0
        for word in topic:
            tf = word_tf[thread_id].get(word)
            if tf:
                num_word += 1
                tfidf += tf * math.log(len(threads)/word_df[word])
        if num_word != 0:
            tfidf = tfidf / num_word
        topic_tfidf[thread_id].append(tfidf)
                
                
                
    

topic_tfidf

topic_distribution = [0] * len(topics)
for thread_id, topic_ifidf_lst in topic_tfidf.items():
    for i in range(len(topic_ifidf_lst)):
        if topic_ifidf_lst[i] != 0:
            topic_distribution[i] += 1

topic_distribution

[len(t) for t in topics]

import math

topic_tfidf = {}
for thread_id in word_tf:
    topic_tfidf[thread_id] = []
    for topic in topics:
#         print(topic)
        tfidf = 0
        num_word = 0
        for word in topic:
            tf = word_tf[thread_id].get(word)
            if tf:
                num_word += 1
                tfidf += math.log(tf * math.log(len(threads)/word_df[word]))
        if num_word != 0:
            tfidf = tfidf/num_word
        topic_tfidf[thread_id].append(tfidf)

topic_distribution = [0] * len(topics)
for thread_id, topic_ifidf_lst in topic_tfidf.items():
    for i in range(len(topic_ifidf_lst)):
        if topic_ifidf_lst[i] != 0:
            topic_distribution[i] += 1

topic_distribution

[1611, 99, 2645, 2366, 1308, 3306, 856, 2330, 1562]

topic_tfidf = []
for topic in topics:
    topic_tf = 0
    topic_df = 0
    for word in topic:
        # calculate sum of word tf
        for thread_id, value in word_tf.items():
            if word in value:
                topic_tf += value[word]
        # calculate sum of word df
        topic_df += word_df.get(word, 0)
    topic_ti = topic_tf/topic_df
    topic_tfidf.append(topic_ti)

topic_tfidf



