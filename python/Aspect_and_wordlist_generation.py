from pymongo import MongoClient, ASCENDING
from srs.database import connect_to_db
from srs.utilities import Sentence, tokenize
from nltk import pos_tag
from collections import Counter
import math
import word2vec
import os
import numpy as np
import random
import copy
import gzip
import ast
import operator
# Loading Word2Vec model:
current_directory = os.path.dirname(os.path.realpath("__file__"))
model_path = os.path.join(current_directory[:-6], 'srs/predictor_data/text8.bin')
model = word2vec.load(model_path)

# Define some generally used functions:
def sort_list(list, sort_index, reverse = True):
    list_sorted = sorted(list, key=lambda tup: tup[sort_index], reverse = reverse)
    return list_sorted

def get_excluded_words():
    f = open("Aspect_and_wordlist_txt/excluded_words.txt",'r')
    excluded_words = eval(f.read())
    f.close()
    return excluded_words

def get_excluded_words_wordlist(category_id):
    f = open("Aspect_and_wordlist_txt/excluded_words_wordlist.txt",'r')
    dictionary = eval(f.read())
    if category_id in dictionary:
        excluded_words_dict = dictionary[category_id]
        is_apply_all = int(excluded_words_dict["apply_all"])
        f.close()
        return is_apply_all, excluded_words_dict
    else:
        f.close()
        return -1, {}

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield ast.literal_eval(l)


def construct_prod_dict(meta_file_path_list):
    """return a dictionary for product metadata"""
    prod_dict = {}
    for meta_file_path in meta_file_path_list:
        metaParser = parse(meta_file_path)
        client, db = connect_to_db()
        i = 0       
        print "Building the product dictionary for %s" % meta_file_path
        for meta in metaParser:
            i+=1
            if i % 100000 == 0:
                print i
            product_id = meta['asin']
            category = meta['categories'][0]
            product_name = ""
            brand = ""
            if 'title' in meta:
                inter = meta['title'].split()
                if len (inter) > 1:
                    product_name_short = inter[0] + ' ' + inter[1]
                else:
                    product_name_short = inter[0]
            if 'brand' in meta:
                brand = meta['brand']
            prod_dict[product_id]={'category': category, 'product_name': product_name_short, 'brand': brand}
        print i
    return prod_dict

Electronics_Meta_Path = '../../Datasets/Full_Reviews/meta_Electronics.json.gz'
Phone_Meta_Path = '../../Datasets/Full_Reviews/meta_Cell_Phones_and_Accessories.json.gz'

prod_dict = construct_prod_dict([Electronics_Meta_Path,Phone_Meta_Path])

def get_category_dict(prod_dict):
    """Build a dictionary whose key is the category tuple, and the value is a list of product_ids:"""
    client, db = connect_to_db()
    cursor = db.product_collection.find()
    category_dict = {}
    i = 0
    for product in cursor:
        i += 1   
        if i % 100000 == 0:
            print i
        category = product['category']
        category_short = tuple(category[:4]) #generally category is 4-tuple. Now limit to the first three tuple
        product_id = product['product_id']
        product_name = ""
        brand = ""
        if product_id in prod_dict:
            product_info = prod_dict[product_id]
            if 'product_name' in product_info:
                product_name = product_info['product_name']
            if 'brand' in product_info:
                brand = product_info['brand']

        if category_short not in category_dict:
            category_dict[category_short] = {"product_id": [product_id], "brand_list": [], "product_name_list": []}
        else:
            category_dict[category_short]['product_id'].append(product_id)
            
        if len(product_name) > 0:
            category_dict[category_short]['product_name_list'].append(product_name)
        if len(brand) > 0:
            if brand not in category_dict[category_short]['brand_list']:
                category_dict[category_short]['brand_list'].append(brand)
            
    client.close()
    print i
  
    return category_dict


def sort_category_dict(category_dict, isPrint = False):
    """Sort the categories according to the number of products in that category, and print them from top"""
    category_list_sorted = []
    category_list = []

    for key in category_dict:
        length = len(category_dict[key]['product_id'])
        category_list.append([key,length,key[:3],0])
    category_list_sorted = sorted(category_list, key=lambda tup: (tup[2],tup[1]), reverse=True)
    
    category_list_sorted_dict = {}
    for Id in range(len(category_list_sorted)):
        category_list_sorted[Id][3]=Id
        category = category_list_sorted[Id][0]
        category_dict[category]["category_id"] = Id
        category_list_sorted_dict[Id] = category_list_sorted[Id][:3]
    
    if isPrint:
        for Id in range(len(category_list_sorted)):
            print Id, category_list_sorted_dict[Id][:2]
        
    return category_list_sorted_dict


def combine_category_custom(category_dict_raw, category_list_sorted_dict):
    category_dict = copy.deepcopy(category_dict_raw)
    print "Number of categories in original set: %g"%len(category_dict_raw)
    print "Combined category ID:"
    f = open('Aspect_and_wordlist_txt/combined_dict.txt','r')
    for line in f:
        combine_info = eval(line)
        print combine_info
        if len(combine_info) > 0:
            Id_to_combine = combine_info[0]
            name_info = combine_info[1]
            category_name_combined = category_list_sorted_dict[name_info[0]][0][:name_info[1]]
            category_id = category_dict_raw[category_list_sorted_dict[name_info[0]][0]]["category_id"]
            new_prod_id_list = []
            new_product_name_list = []
            new_brand_list = []
            for Id in Id_to_combine:
                category_name = category_list_sorted_dict[Id][0]
                new_prod_id_list += category_dict[category_name]["product_id"]
                new_product_name_list += category_dict[category_name]["product_name_list"]
                new_brand_list += category_dict[category_name]["brand_list"]
                category_dict.pop(category_name, 0)
            category_dict[category_name_combined] = {"category_id": category_id,"product_id": new_prod_id_list,                        "product_name_list": new_product_name_list, "brand_list": new_brand_list}
    f.close()
    print "Number of categories in the new dict: %g"%len(category_dict)
      
    return category_dict


def combine_small_category(category_dict_raw, category_list_sorted, prod_num_threshold = 100, shrink_level = 3):
    category_dict = copy.deepcopy(category_dict_raw)
    i = 0
    for i in range(len(category_list_sorted)):
        i += 1
        category_name = category_list_sorted[-i][1]
        prod_num = category_list_sorted[-i][0]
        if prod_num > prod_num_threshold:
            break
        if len(category_name) > shrink_level:
            category_name_shrink = category_name[:shrink_level]
            if category_name_shrink in category_dict:
                category_dict[category_name_shrink] += category_dict[category_name]
                category_dict.pop(category_name,0)
                print "{0} combined into {1}".format(category_name_shrink, category_name)
            else:
                print "{0} not combined".format(category_name_shrink)
        else:
            print "{0} length not enough.".format(category_name)
    
    return category_dict


def save_category_dict_to_db(category_dict, dropPrevious = False):
    client, db = connect_to_db()
    db_category_data = db.category_data
    if dropPrevious == True:
        db_category_data.delete_many({})
    for category in category_dict:
        query = {"category_id": category_dict[category]["category_id"]}
        update_field = {"category": list(category),                        "prod_id_list": category_dict[category]["product_id"],                         "brand_list":  category_dict[category]["brand_list"],                        "product_name_list": category_dict[category]["product_name_list"]}
        db_category_data.update_one(query, {"$set": update_field}, True)
        
    client.close()


def show_category_dict_info(category_dict, min_prod_num = 1000):
    new_list = []
    for category in category_dict:
        new_list.append([len(category_dict[category]["product_id"]),category,category_dict[category]["category_id"]])
    
    new_list = sorted(new_list, key=lambda tup: tup[0], reverse=True)
    
    for item in new_list:
        if int(item[0]) < min_prod_num:        
            break
        print "{0},{1},{2}".format(item[0],item[1],item[2])


def get_sentence_from_category(category_list):
    """Obtain all the review sentences from a list of category tuple:"""
    if isinstance(category_list, dict):
        category_lists = [category_list]
    else:
        category_lists = category_list
    
    category_content_list = []
    
    for category in category_lists:
        print "{0}:".format(category)
        client, db = connect_to_db()
        product_id_list = category_dict[category]["product_id"]
        category_contents = {"category": category,"sentence_list": [], "brand_list": category_dict[category]["brand_list"],                            "product_name_list": category_dict[category]["product_name_list"]}
        review_num = 0
        for product_id in product_id_list:
            query_res = list(db.product_collection.find({"product_id": product_id}))
            contents = query_res[0]["contents"]
            category_contents['sentence_list'] += contents
            review_num += len(query_res[0]["review_ids"])
        print "  ({0}, {1}, {2})".format(len(product_id_list), review_num, len(category_contents['sentence_list']))      
        category_content_list.append(category_contents)
        
    client.close()

    return category_content_list


def get_sentence_from_category_ensemble(category_dict, max_prod_chosen = 500, min_product_level = 500):
    client, db = connect_to_db()
    full_sentence_list = []
    print "Getting product categories: (num_sentence_chosen, category):"
    for category in category_dict:
        if len(category_dict[category]) < min_product_level:
            continue
        product_id_list = category_dict[category]["product_id"]
        random.shuffle(product_id_list)
        new_sentence = []
        for product_id in product_id_list[:max_prod_chosen]:
            query_res = list(db.product_collection.find({"product_id": product_id}))
            contents = query_res[0]["contents"]
            new_sentence += contents
        print len(new_sentence),category
        full_sentence_list += new_sentence
    client.close()
    print "Number of sentences: {0}".format(len(full_sentence_list))
    
    all_category_content = {"sentence_list": full_sentence_list}
    return all_category_content

category_dict_raw = get_category_dict(prod_dict)

category_list_sorted_dict = sort_category_dict(category_dict_raw, isPrint = False)
category_dict = combine_category_custom(category_dict_raw, category_list_sorted_dict)
save_category_dict_to_db(category_dict, dropPrevious = False)

show_category_dict_info(category_dict, min_prod_num = 1000)

# all_category_content = get_sentence_from_category_ensemble(category_dict, max_prod_chosen = 1000, min_product_level = 0)
# get_tf_idf(all_category_content, is_idf_db = False)

def get_category_word_scores(category_id_list, db_category_data = None, db_product_collection = None, db_word_score_list = None):
    """Get tf-idf score for each word
       The dictionary records for each word as a key, the [num_word, num_doc] value, where num_word means the number of 
       that word in the sentence_list, and num_doc means the number of sentences this word appears in.
    """
    external_db = True
    if (not db_category_data) or (not db_product_collection) or (not db_word_score_list):
        external_db = False
        client, db = connect_to_db()
        if not db_category_data:
            db_category_data = db.category_data
        if not db_product_collection:
            db_product_collection = db.product_collection
        if not db_word_score_list:
            db_word_score_list = db.word_score_list
    
    if not isinstance(category_id_list, list):
        category_id_list = [category_id_list]
    
    # Collecting each word's data for that category
    for category_id in category_id_list:
        query_category = list(db_category_data.find({"category_id": category_id}))
        if len(query_category) == 0:
            print "{0} not in db, skip.".format(category_id)
            continue
        category_content = query_category[0]
        category = category_content["category"]
        prod_id_list = category_content["prod_id_list"]
        prod_num = len(prod_id_list)     
        
        # Obtaining brand_list words and product_name words:
        brand_list = category_content["brand_list"]
        product_name_list = category_content["product_name_list"]
        brand_word_list = []
        product_name_word_list =[]
        for brand in brand_list:     
            brand_word = tokenize(brand, stem = False)
            if len(brand_word) > 0:
                brand_word_list += brand_word[:1]
        brand_word_list = dict(Counter(brand_word_list))
        for product_name in product_name_list:   
            product_name_word = tokenize(product_name, stem = False)
            if len(product_name_word) > 0:
                product_name_word_list += product_name_word[:1]
        product_name_word_list = dict(Counter(product_name_word_list))
       
        # Obtaining word_statistics: [word, word_freq, num_doc]
        
        word_statistics = {}       
        i = 0
        for product_id in prod_id_list:          
            query_res = list(db_product_collection.find({"product_id": product_id}))
            contents = query_res[0]["contents"]
            for sentence in contents:
                i += 1
                if i % 100000 == 0:
                    print i
                tokens = tokenize(sentence, stem = False)
                tokens_count = Counter(tokens)
                for word in tokens_count:        
                    if word not in word_statistics:
                        word_statistics[word] = [tokens_count[word], 1]
                    else:
                        word_statistics[word][0] += tokens_count[word]
                        word_statistics[word][1] += 1
        
        total_num_doc = i
        print "Id: {0}, num_prod: {1}, num_sentence: {2}".format(category_id, prod_num, total_num_doc)
        print "{0}".format(category)
        word_scores = []
        
        max_word_freq = 0
        for word in word_statistics:
            if word_statistics[word][0] > max_word_freq:
                max_word_freq = word_statistics[word][0]
        
        # Calculating tf-idf for the category
        for word in word_statistics:               
            word_rawdata = word_statistics[word]
            word_freq = word_rawdata[0]
            num_doc = word_rawdata[1]
            tf = float(word_freq) / max_word_freq 
                                   
            idf_category = math.log(float(total_num_doc)/(num_doc))
            query_idf = list(db_word_score_list.find({"word": word}))
            if len(query_idf) > 0:
                idf = query_idf[0]["full_word_score"][2]
            else:
                idf = idf_category
            word_scores.append([word, tf * idf, tf, idf_category, idf, word_freq, num_doc])
                                   
        word_statistics.clear()
        word_scores.sort(key=lambda tup: tup[1], reverse=True)
             
        # Update database:
        query = {"category_id": category_id}
        update_field = {"word_scores": word_scores, "brand_word_list": brand_word_list,                        "product_name_word_list": product_name_word_list,"total_num_sentence":total_num_doc}
        db_category_data.update_one(query, {"$set": update_field}, True)
        
        # Get aspect_cadidate:
        if prod_num < 20:
            num_candidate = 80
        else:
            num_candidate = 60
    
    if external_db == False:
        client.close()   



def get_aspect_cadidate(category_id_list, tag_list = ["NN","NNS","JJ"], num_candidate = -1, rescan_word_scores = False):
    '''Get cadidate aspects from word_tf_idf. Only words whose tag belong to tag_list and score > threshold will pass'''  
    # check if db cursor is given:
    client, db = connect_to_db()
    db_word_score_list = db.word_score_list
    db_product_collection = db.product_collection
    db_category_data = db.category_data

    for category_id in category_id_list: 
        full_word_freq_thresh = 30
        full_num_doc_thresh = 10
        
        words_excluded = []
        query_category = list(db_category_data.find({"category_id": category_id}))
        if len(query_category) == 0:
            print "{0} not in db, skip.".format(category_id)
            continue
        category_content = query_category[0]
        category = category_content["category"]
        num_prod = len(category_content["prod_id_list"])
        if num_prod < 10:
            continue
        print "{0}, {1}".format(category_id, category)
        # Check if need to rescanning word_scores:
        reQuery = False
        if "word_scores" not in category_content or "word_scores" not in category_content or "product_name_word_list" not in category_content:
            print "{0} don't have word_scores, constructing...".format(category_id)
            get_category_word_scores(category_id, db_category_data, db_product_collection, db_word_score_list)
            reQuery = True
        elif rescan_word_scores == True:
            print "rescanning word_scores...".format(category_id)
            get_category_word_scores(category_id, db_category_data, db_product_collection, db_word_score_list)
            reQuery = True
        
        if reQuery == True:
            category_content = list(db_category_data.find({"category_id": category_id}))[0]
        word_scores = category_content["word_scores"]
        brand_word_list = category_content["brand_word_list"]
        product_name_word_list = category_content["product_name_word_list"]
           
        
        #Setting num_candidate
        if num_candidate == -1:
            if num_prod >= 20:
                num_candidate2 = 70
            else:
                num_candidate2 = 50
        else:
            num_candidate2 = num_candidate
               
        if num_prod < 50:
            full_word_freq_thresh = int(num_prod/ 4)
            full_num_doc_thresh = int(num_prod / 8)
                    
        aspect_candidate = []
        j = 0
        try:
            excluded_words = get_excluded_words()
        except:
            print "Problem with excluded_words.txt, use previous."
        for word_data in word_scores:
            full_word_freq = 10000
            full_num_doc = 10000 # default setting
            word = word_data[0]
            # Various criterior to exclude the word:
            if len(word) == 1:
                continue
            if word_data[5] <= 1:
                continue
            if word in excluded_words:
                continue
            query_idf = list(db_word_score_list.find({"word": word}))
            if len(query_idf) > 0:
                full_word_score = query_idf[0]["full_word_score"]
                full_word_freq = full_word_score[4]
                full_num_doc = full_word_score[5]
                if full_word_freq < full_word_freq_thresh or full_num_doc < full_num_doc_thresh:
                    words_excluded.append(word)
                    continue            
            if word in brand_word_list:
                if full_word_freq < 100:
                    words_excluded.append(word)
                    continue
            if word in product_name_word_list:
                if full_word_freq < 150:
                    words_excluded.append(word)
                    continue                        
                                  
            word_tag = pos_tag([word])[0][1]
            # If the tag is in tag_list:
            if word_tag in tag_list:
                j += 1
                word_data.append(word_tag)
                aspect_candidate.append(word_data)
                word1 = word
                if len(word1) <= 2:
                    word1 += " "
                print "%s     \t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%g\t%g\t%s"%(word1, word_data[1],word_data[2],word_data[3],word_data[4],                                                               word_data[5], word_data[6], word_data[7])
            else:
                words_excluded.append(word)
            if j > num_candidate2:
                break
               
        # Update database:
        query = {"category_id": category_id}
        update_field = {"aspect_candidate": aspect_candidate, "words_excluded": words_excluded}
        db_category_data.update_one(query, {"$set": update_field}, True)      
        print "Excluded words: {0}".format(words_excluded)
        print
    client.close()


def save_word_score_to_db(category_content_list, isRewrite = False):
    client, db = connect_to_db()
    db_word_score_list = db.word_score_list
    if isRewrite == True:
        db_word_score_list.delete_many({})
        db_word_score_list.create_index([("word", ASCENDING)])
        db_word_score_list.create_index([("category", ASCENDING)])
    
    if isinstance(category_content_list, dict):
        category_content_lists = [category_content_list]
    else:
        category_content_lists = category_content_list
    
    for category_content in category_content_lists:
        word_tf_idf = category_content["word_tf_idf"]
        category = category_content["category"]
        i = 0
        for word_data in word_tf_idf:
            i += 1
            if i % 50000 == 0:
                print i
            word = word_data[0]
            word_score = word_data[1:]
            query = {"word": word}
            update_field = {"category": category, "word_score": word_score}

            db_word_score_list.update_one(query, {"$set": update_field}, True)       
        print "{0}: Total number of words: {1}".format(category, len(word_tf_idf))
    
    client.close()

def get_similarity(word1, word2):
    """Find the similarity between two words, which equals the dot product of their vectors"""
    similarity = 0
    word1=word1.lower()
    word2=word2.lower()
    if word1 in model and word2 in model:
        word1_vec = model[word1]
        word2_vec = model[word2]
        similarity = np.dot(word1_vec, word2_vec)
    return similarity

def get_wordlist_from_aspect_candidates(seed_word, word_tf_idf, similarity_threshold, score_threshold):
    """Method 1: directly find the word list from all words whose similarity with the seed_word and tf-idf score are above 
    certain threshold"""
    word_list = []
    for word_data in word_tf_idf:
        word = word_data[0]
        tf_idf = word_data[1]
        if tf_idf > score_threshold:
            similarity = get_similarity(seed_word, word)
            if similarity > similarity_threshold:
                word_list.append([word, similarity, tf_idf])              
    word_list_sorted = sorted(word_list, key=lambda tup: tup[1], reverse=True)
    return word_list_sorted


def predict_aspect(token_list, wordlist_dict, predict_threshold = 1.05):
    """
    sentence: a single labelled sentence obj 
    returns a vector of length len(wordlist_dict)
    """
    
    len_word = max(len(token_list),1) * 1.0;
    f_vec = copy.deepcopy(wordlist_dict) #speed up by putting copy and reset outside
    #reset to zero 
    for key in f_vec.keys():
        for i in range(len(f_vec[key])):
            f_vec[key][i][1]=0

    for key in wordlist_dict.keys():
        for i in range(len(wordlist_dict[key])):
            count = token_list.count(wordlist_dict[key][i][0])
            f_vec[key][i][1]=(count/len_word)
    
    #multiply it with weights
    score_dict = dict.fromkeys(f_vec, 0)
    for key in f_vec.keys():
        dot_product = 0.0
        for i in range(len(wordlist_dict[key])):
            dot_product += np.exp(wordlist_dict[key][i][1]*f_vec[key][i][1])

        score_dict[key] = dot_product/len(wordlist_dict[key]) # min score is 1
    
    predicted_aspect = max(score_dict.iteritems(), key = operator.itemgetter(1))[0]
    if score_dict[predicted_aspect] <= predict_threshold: # no key word overlap
        #call word2vec similarity 
        # score_dict_w2v = word2vec_predict(sentence,wordlist_dict,w2v_model)
        # predicted_aspect = max(score_dict_w2v.iteritems(), key=operator.itemgetter(1))[0]
        # if score_dict_w2v[predicted_aspect] < thres:
        predicted_aspect = "no feature"

    return predicted_aspect


def test_prediction(category_id, wordlist_dict, aspect_to_show = [], predict_threshold = 1.05, num_show = 100, show_no_feature = False):
    client, db = connect_to_db()
    db_category_data = db.category_data
    db_product_collection = db.product_collection
    query_category = list(db_category_data.find({"category_id": category_id}))
    if len(query_category) == 0:
        print "category {0} not in db.".format(category_id)
        return
    category_content = query_category[0]
    category = category_content["category"]
    prod_id_list = category_content["prod_id_list"]
    random.shuffle(prod_id_list)
    num_prod = len(prod_id_list)
    
    i = 0
    for prod_id in prod_id_list:
        query_product = list(db_product_collection.find({"product_id": prod_id}))
        if len(query_product) == 0:
            print "product {0} not in db, skip.".format(prod_id)
        sentence_list = query_product[0]["contents"]
        random.shuffle(sentence_list)
        for sentence in sentence_list[:max(30,num_show / num_prod * 5)]:            
            if i > num_show:
                break
            token_list = tokenize(sentence, stem = False)
            predicted_aspect = predict_aspect(token_list, wordlist_dict, predict_threshold)
            if show_no_feature == False:
                if predicted_aspect == "no feature":
                    continue
            i += 1
            if len(aspect_to_show) == 0:
                print "{0}:\t{1}".format(predicted_aspect, sentence)
            else:
                if predicted_aspect in aspect_to_show:
                    print "{0}:\t{1}".format(predicted_aspect, sentence)
    client.close()      



def get_word_statistics_from_seed_word(prod_id_list, seed_word_list):
    client, db = connect_to_db()
    db_product_collection = db.product_collection
    word_statistics_dict = {seed_word.lower(): {} for seed_word in seed_word_list}
    aspect_sentence_num_dict = {seed_word.lower(): 0 for seed_word in seed_word_list}
    aspect_sentence_num_dict["no feature"] = 0
    sentence_num = 0
    for product_id in prod_id_list:
        query_product = list(db_product_collection.find({"product_id": product_id}))
        if len(query_product) == 0:
            print "Product {0} doesn't exists, skip.".format(product_id)
            continue
        if "contents" not in query_product[0]:
            print "Product {0} doesn't have contents, skip.".format(product_id)
            continue
        sentence_list = query_product[0]["contents"]
        for sentence in sentence_list:  
            tokens_list = tokenize(sentence, stem = False)                
            tokens_count = Counter(tokens_list)
            sentence_num += 1
            if sentence_num % 100000 == 0:
                print sentence_num
            has_feature = 0
            for seed_word in word_statistics_dict:                
                if seed_word in tokens_count:
                    has_feature = 1
                    aspect_sentence_num_dict[seed_word] += 1
                    for token in tokens_count:
                        if token in word_statistics_dict[seed_word]:
                            word_statistics_dict[seed_word][token][0] += tokens_count[token]
                            word_statistics_dict[seed_word][token][1] += 1
                        else:
                            word_statistics_dict[seed_word][token] = [tokens_count[token], 1, 0, 0, 0, 0, 0, 0]
            if has_feature == 0:
                aspect_sentence_num_dict["no feature"] += 1
    for seed_word in word_statistics_dict:
        word_statistics_tuple_sorted = sorted(word_statistics_dict[seed_word].items(), key=operator.itemgetter(1), reverse = True)
        word_statistics_list_sorted = []
        
        for i in range(len(word_statistics_tuple_sorted)):
            word_statistics_list_sorted.append(list(word_statistics_tuple_sorted[i]))
    
        # Calculate tf-idf:
        try:
            max_term_freq = word_statistics_list_sorted[0][1][0]
        except:
            print "cannot access the first element"
            print seed_word, word_statistics_list_sorted, word_statistics_tuple_sorted
            max_term_freq = 1
        for i in range(len(word_statistics_list_sorted)):
            word_data = word_statistics_list_sorted[i][1]
            tf_aspect = float(word_data[0]) / max_term_freq
            idf_aspect = math.log(float(sentence_num) / word_data[1])
            word_data[2:4] =[tf_aspect, idf_aspect]       
        word_statistics_dict[seed_word] = word_statistics_list_sorted
    
    client.close()
    return word_statistics_dict, aspect_sentence_num_dict
        
    

def get_word_statistics_from_wordlist_dict(prod_id_list, wordlist_dict, predict_threshold = 1):
    client, db = connect_to_db()
    db_product_collection = db.product_collection
    word_statistics_dict = {key: {} for key in wordlist_dict}
    aspect_sentence_num_dict = {aspect.lower(): 0 for aspect in wordlist_dict}
    aspect_sentence_num_dict["no feature"] = 0
    sentence_num = 0
    for product_id in prod_id_list:
        query_product = list(db_product_collection.find({"product_id": product_id}))
        if len(query_product) == 0:
            print "Product {0} doesn't exists, skip.".format(product_id)
            continue
        if "contents" not in query_product[0]:
            print "Product {0} doesn't have contents, skip.".format(product_id)
            continue
        sentence_list = query_product[0]["contents"]
        for sentence in sentence_list:  
            tokens = tokenize(sentence, stem = False)                
            predicted_aspect = predict_aspect(tokens, wordlist_dict, predict_threshold)
            if predicted_aspect == "no feature":
                aspect_sentence_num_dict["no feature"] += 1
                continue
            tokens_count = Counter(tokens)
            aspect_sentence_num_dict[predicted_aspect] += 1
            sentence_num += 1            
            if int(sentence_num) % 100000 == 0:
                print "{0}, {1}: {2}".format(sentence_num, predicted_aspect, sentence)
            for token in tokens_count:
                # Constructing aspect word_tf_idf_dict:                            
                if token in word_statistics_dict[predicted_aspect]:
                    word_statistics_dict[predicted_aspect][token][0] += tokens_count[token]
                    word_statistics_dict[predicted_aspect][token][1] += 1
                else:
                    word_statistics_dict[predicted_aspect][token] = [tokens_count[token], 1, 0, 0, 0, 0, 0, 0]
    
    for aspect in word_statistics_dict:
        word_statistics_tuple_sorted = sorted(word_statistics_dict[aspect].items(), key=operator.itemgetter(1), reverse = True)
        word_statistics_list_sorted = []
        
        for i in range(len(word_statistics_tuple_sorted)):
            word_statistics_list_sorted.append(list(word_statistics_tuple_sorted[i]))
    
        # Calculate tf-idf:
        try:
            max_term_freq = word_statistics_list_sorted[0][1][0]
        except:
            print "cannot access the first element"
            print word_statistics_list_sorted[0]
            max_term_freq = 1
            
        for i in range(len(word_statistics_list_sorted)):
            word_data = word_statistics_list_sorted[i][1]
            tf_aspect = float(word_data[0]) / max_term_freq
            idf_aspect = math.log(float(sentence_num) / word_data[1])
            word_data[2:4] =[tf_aspect, idf_aspect]       
        word_statistics_dict[aspect] = word_statistics_list_sorted
    
    client.close()
    return word_statistics_dict, aspect_sentence_num_dict


def get_wordlist_dict(category_id_list, wordlist_dict, num_words_in_wordlist = 10, sim_slope = 1, sim_intercept = 0.2, predict_threshold = 1, isPrint = True):
    client, db = connect_to_db()
    db_category_data = db.category_data
    if not isinstance(category_id_list, list):
        isList = False
        category_id_list = [category_id_list]
    if isinstance(wordlist_dict, list):
        input_type = "seed_word_list"
    elif isinstance(wordlist_dict, dict):
        input_type = "wordlist_dict"
    else:
        print "please input a seed_word_list or wordlist_dict!"
        return
      
    for category_id in category_id_list:
        query_category = list(db_category_data.find({"category_id": category_id}))
        if len(query_category) == 0:
            print "{0} not in db, skip.".format(category_id)
            continue
        category_content = query_category[0]
        category = category_content["category"]
        prod_id_list = category_content["prod_id_list"]
        
        if "word_scores" not in category_content or "total_num_sentence" not in category_content:
            print "word_scores not in category {0}, constructing...".format(category_id)
            get_category_word_scores(category_id)
        word_scores_list = category_content["word_scores"]
        word_scores_dict = {word_data[0]: word_data[1:] for word_data in word_scores_list}
        total_num_sentence = category_content["total_num_sentence"]
        if input_type == "seed_word_list":
            word_statistics_dict, aspect_sentence_num_dict = get_word_statistics_from_seed_word(prod_id_list, seed_word_list)
        elif input_type == "wordlist_dict":
            word_statistics_dict, aspect_sentence_num_dict = get_word_statistics_from_wordlist_dict(prod_id_list, wordlist_dict, predict_threshold)
        total_prod_num = 0
        for aspect in aspect_sentence_num_dict:
            total_prod_num += aspect_sentence_num_dict[aspect]
        print "total sentence num: {0}".format(total_prod_num)
        for aspect in word_statistics_dict:
            word_statistics = word_statistics_dict[aspect]
            aspect_sentence_num = aspect_sentence_num_dict[aspect]
            print "{0}: {1}".format(aspect, aspect_sentence_num)
            for word_data in word_statistics:
                
                word = word_data[0]
                tf_aspect = word_data[1][2]
                idf_category = word_scores_dict[word][2]
                tf_ratio = (float(word_data[1][0]) / aspect_sentence_num) / (float(word_scores_dict[word][4]) / total_num_sentence)
                similarity = get_similarity(aspect, word)
                
                word_data[1][4] = idf_category
                word_data[1][5] = tf_ratio
                word_data[1][6] = similarity
                word_data[1][7] = math.log(1 +  tf_aspect ** 0.8 * idf_category **2  *  max(0.1, math.log(tf_ratio)) * max(0.01, sim_slope * similarity + sim_intercept)** 1.5)
                    
            word_statistics.sort(key = lambda tup: tup[1][7], reverse = True)
        print "no feature: {0}".format(aspect_sentence_num_dict["no feature"])
        wordlist_dict = {}
        for aspect in word_statistics_dict:
            wordlist = []
            for word_data in word_statistics_dict[aspect][:num_words_in_wordlist]:
                wordlist.append([word_data[0],word_data[1][7]])
            wordlist_dict[aspect] = wordlist
        

        update_field = {"wordlist_dict": wordlist_dict}
        db_category_data.update_one({"category_id": category_id}, {"$set": update_field}, True)
        try:
            update_field = {"word_statistics_dict": word_statistics_dict}
            db_category_data.update_one({"category_id": category_id}, {"$set": update_field}, True)
        except:
            print "word_statistics too large, cannot save into db, skip."
        
    if isPrint == True:
        for aspect in wordlist_dict:
            print '"{0}": '.format(aspect)
            print '  ',
            for word_data in wordlist_dict[aspect]:
                print '"%s", %0.2f;'%(word_data[0], word_data[1]),
            print
    client.close()
    return wordlist_dict


def prune_wordlist_dict(wordlist_dict, excluded_word_external = [], preserve_top = False, isPrint = True):
    word_location = {}
    excluded_words = ["good", "great", "well", "bad", "worse","better"] + excluded_word_external
    # Obtaining each word's aspects and score in that aspect:   
    if preserve_top == True:
        for aspect in wordlist_dict:
            for word_data in wordlist_dict[aspect]:
                word = word_data[0]
                score = word_data[1]
                if word in word_location:
                    word_location[word].append([aspect, score])
                else:
                    word_location[word]=[[aspect, score]]             

        # Sort each word's aspect, and only keep the word in highest score aspect:   
        wordlist_dict_pruned = {}
        for word in word_location:
            if word in excluded_words:
                continue
            aspect_sorted = sort_list(word_location[word], 1)
            aspect_chosen = aspect_sorted[0] # Choose the first one
            aspect = aspect_chosen[0]
            if aspect in wordlist_dict_pruned:
                wordlist_dict_pruned[aspect].append([word, aspect_chosen[1]])
            else:
                wordlist_dict_pruned[aspect] = [[word, aspect_chosen[1]]]
    else:
        wordlist_dict_pruned = {}
        for aspect in wordlist_dict:
            wordlist = []
            for word in wordlist_dict[aspect]:
                if word[0] not in excluded_words:
                    wordlist.append(word) 
            wordlist_dict_pruned[aspect] = wordlist
    
    # Sort each word
    for word in wordlist_dict_pruned:
        wordlist = wordlist_dict_pruned[word]
        wordlist_dict_pruned[word] = sort_list(wordlist, 1)
    
    if isPrint == True:
        for aspect in wordlist_dict_pruned:
            print '"{0}": '.format(aspect)
            print '  ',
            for word_data in wordlist_dict_pruned[aspect]:
                print '"%s", %0.2f;'%(word_data[0], word_data[1]),
            print
        
    return wordlist_dict_pruned


def get_category_data_from_db(category_id, request_field_list):
    client, db = connect_to_db()
    db_category_data = db.category_data
    result = []
    if not isinstance(request_field_list, list):
        request_field_list = [request_field_list]
        isOnefield = True
    query = list(db_category_data.find({"category_id": category_id}))
    if len(query) > 0:
        query = query[0]
        for request_field in request_field_list:        
            if request_field in query:
                result.append(query[request_field])
            else:
                print "{0} not in category {1}".format(request_field, category_id)
                result.append([])
    else:
        print "category {0} not in db".format(category_id)
        result = [[] for request_field in request_field_list]
    if isOnefield == True:
        result = result[0]
    return result
                            

def writeWordlistDictToDB(category_id, wordlist_dict, rewrite_wordlist_dict_list = False):
    # Update database:    
    client, db = connect_to_db()
    query = {"category_id": category_id}
    query_res = list(db.category_data.find(query))
    if len(query_res) > 0:
        if "wordlist_dict_list" in query_res[0] and rewrite_wordlist_dict_list == False:
            wordlist_dict_list = query_res[0]["wordlist_dict_list"]
            wordlist_dict_list.append(wordlist_dict)
        else:
            wordlist_dict_list = []
    else:
        wordlist_dict_list = []
    wordlist_dict_list.append(wordlist_dict)   
    
    update_field = {"wordlist_dict": wordlist_dict, "wordlist_dict_list": wordlist_dict_list}
    db.category_data.update_one(query, {"$set": update_field}, True)
    category = list(db.category_data.find({"category_id": category_id}))[0]["category"]
    
    query_category = {"category_id": category_id}
    update_field_category = {"category": category, "wordlist_dict": wordlist_dict}
    db.category_collection.update_one(query_category, {"$set": update_field_category}, True)
    client.close()


def changeAspectNameinDB(category_id, aspect_old_name, aspect_new_name):
    client, db = connect_to_db()
    change_aspect_name_in_db(category_id, aspect_old_name, aspect_new_name, db.category_data)
    change_aspect_name_in_db(category_id, aspect_old_name, aspect_new_name, db.category_collection)
    client.close()   
    
def change_aspect_name_in_db(category_id, aspect_old_name, aspect_new_name, db_collection):
    query = {"category_id": category_id}
    query_res = list(db_collection.find(query))
    if len(query_res) == 0:
        print "Category {0} do not exist!".format(category_id)
        return
    category_content = query_res[0]
    if "wordlist_dict" not in category_content:
        print "Category {0} do not have wordlist_dict!".format(category_id)
        return
    wordlist_dict = category_content["wordlist_dict"]
    if aspect_old_name not in wordlist_dict:
        print 'Category {0} do not have aspect "{1}"'.format(category_id, aspect_old_name)
        return
    wordlist_dict[aspect_new_name] = wordlist_dict.pop(aspect_old_name)
    
    #update db:
    db_collection.update_one(query,{"$set": {"wordlist_dict": wordlist_dict}},False)
    print "Changing aspect name successful :D"

    
def addWordtoAspectinDB(category_id, word_data, aspect_to_update):
    client, db = connect_to_db()
    add_word_to_aspect(category_id, word_data, aspect_to_update, db.category_data)
    add_word_to_aspect(category_id, word_data, aspect_to_update, db.category_collection)
    client.close()  

def add_word_to_aspect(category_id, word_data, aspect_to_update, db_collection):
    query = {"category_id": category_id}
    query_res = list(db_collection.find(query))
    if len(query_res) == 0:
        print "Category {0} do not exist!".format(category_id)
        return
    category_content = query_res[0]
    if "wordlist_dict" not in category_content:
        print "Category {0} do not have wordlist_dict!".format(category_id)
        return
    wordlist_dict = category_content["wordlist_dict"]
    if aspect_to_update not in wordlist_dict:
        print 'Category {0} do not have aspect "{1}"'.format(category_id, aspect_old_name)
        return
    
    wordlist_dict[aspect_to_update].append(word_data)
    wordlist_dict[aspect_to_update].sort(key = lambda tup: tup[1], reverse = True)
    
    db_collection.update_one(query,{"$set": {"wordlist_dict": wordlist_dict}},False)
    print 'Successfully add word {0} to category {1}\'s aspect "{2}"'.format(word_data, aspect_to_update, category_id)

    
def moveWordinAspect(category_id, word, previous_aspect, new_aspect):
    client, db = connect_to_db()
    move_word_in_aspect(category_id, word, previous_aspect, new_aspect, db.category_data)
    move_word_in_aspect(category_id, word, previous_aspect, new_aspect, db.category_collection)
    client.close()

def move_word_in_aspect(category_id, word, previous_aspect, new_aspect, db_collection):
    query = {"category_id": category_id}
    query_res = list(db_collection.find(query))
    if len(query_res) == 0:
        print "Category {0} do not exist!".format(category_id)
        return
    category_content = query_res[0]
    if "wordlist_dict" not in category_content:
        print "Category {0} do not have wordlist_dict!".format(category_id)
        return
    wordlist_dict = category_content["wordlist_dict"]
    if previous_aspect not in wordlist_dict:
        print 'Old aspect "{0}" not exists in category {1}'.format(previous_aspect, category_id)
        return
    if new_aspect not in wordlist_dict:
        print 'New aspect "{0}" not exists in category {1}'.format(new_aspect, category_id)
        return
    source_list = wordlist_dict[previous_aspect]
    target_list = wordlist_dict[new_aspect]
    isFound = False
    new_source_list = []
    for word_data in source_list:
        if word_data[0] == word:
            record = word_data
            isFound = True
        else:
            new_source_list.append(word_data)
    if isFound == False:
        print "Word {0} not found in aspect {1}".format(word, previous_aspect)
        return
    target_list.append(record)
    target_list.sort(key = lambda tup: tup[1], reverse = True)
    wordlist_dict[previous_aspect] = new_source_list
    wordlist_dict[new_aspect] = target_list
    db_collection.update_one(query,{"$set": {"wordlist_dict": wordlist_dict}},False)
    print 'Succesfully move word "{0}" from "{1}" to "{2}".'.format(word, previous_aspect, new_aspect)
 
        
def deleteWordsFromDB(category_id_list, words_to_delete, aspects_to_update = []):
    client, db = connect_to_db()
    delete_words_from_collection(category_id_list, words_to_delete, db.category_collection, aspects_to_update)
    delete_words_from_collection(category_id_list, words_to_delete, db.category_data, aspects_to_update)
    client.close()


def delete_words_from_collection(category_id_list, words_to_delete,  db_collection, aspects_to_update = []):
    if not isinstance(words_to_delete, list):
        words_to_delete = [words_to_delete]
    if not isinstance(category_id_list, list):
        category_id_list = [category_id_list]
    if not isinstance(aspects_to_update, list):
        aspects_to_update = [aspects_to_update]
    for category_id in category_id_list:
        query_collection = list(db_collection.find({"category_id": category_id}))
        if len(query_collection) == 0:
            print "{0} not in db, skip".format(category_id)
            continue
        wordlist_dict = query_collection[0]["wordlist_dict"]
        isFound = False
        if len(aspects_to_update) > 0:
            for aspect in aspects_to_update:
                new_wordlist = []
                for word_data in wordlist_dict[aspect]:
                    if word_data[0] not in words_to_delete:
                        new_wordlist.append(word_data)
                    else:
                        isFound = True
                wordlist_dict[aspect] = new_wordlist
        else:
            for aspect in wordlist_dict:
                new_wordlist = []
                for word_data in wordlist_dict[aspect]:
                    if word_data[0] not in words_to_delete:
                        new_wordlist.append(word_data)
                    else:
                        isFound = True
                wordlist_dict[aspect] = new_wordlist
        if isFound == False:
            print "words not found in the designated aspect"
            return
        query_collection = {"category_id": category_id}
        update_field = {"wordlist_dict": wordlist_dict}
        db_collection.update_one(query_collection, {"$set": update_field}, False)
    print "Delete sucessful."

category_id_list = [137, 73, 433, 399, 153, 297, 308, 136, 22, 154, 174, 187, 402, 176, 90, 253]
category_seedword_dict = {
    137: {
        "category": [u'Electronics', u'Computers & Accessories', u'Servers'], \
        "aspect_candidate":["drive","backup","software","network","support","console","storage","install","stability"]
    },
    73:{
        "category": [u'Electronics', u'Portable Audio & Video', u'Portable DVD Players'], \
        "aspect_candidate":["screen", "picture","battery","sound","quality","price","video","size"]
    },
    433:{
        "category": [u'Cell Phones & Accessories', u'Cell Phones'], \
        "aspect_candidate":["screen","battery","camera","sim","call","apps","service","quality","wifi","price","plan","design"]
    },
    399:{
        "category": [u'Electronics', u'Accessories & Supplies', u'Batteries, Chargers & Accessories'], \
        "aspect_candidate":["power","price","quality","plug","protection","adapter"]
    },
    153:{
        "category": [u'Electronics', u'Computers & Accessories', u'Monitors'],\
        "aspect_candidate":["display","color","resolution","price","quality","brightness","contrast","video"]
    },
    297:{
        "category": ["Electronics", "Camera & Photo", "Lenses" ],\
        "aspect_candidate": ["focus","zoom","quality","sensor","macro","price","aperture","sharpness","autofocus"]
    },
    308:{
        "category":[u'Electronics', u'Camera & Photo', u'Digital Cameras'],\
        "aspect_candidate": ["battery","pictures","price","zoom","easy","detection","design","video","quality","screen","size"]
    },
    136:{
        "category": [u'Electronics', u'Computers & Accessories', u'Tablets'],\
        "aspect_candidate": ["battery","screen","wifi","apps","camera","video","gb","touch","quality","price","size"]
    },
    22:{
        "category": [u'Electronics', u'Television & Video', u'Televisions'],\
        "aspect_candidate": ["picture","sound","screen","price","remote","cable","service","audio"]
    },
    154:{
        "category":[u'Electronics', u'Computers & Accessories', u'Laptops'],\
        "aspect_candidate": ["screen","keyboard","battery","drive","price","processor","graphics","touchpad","support"]
    },
    174:{
        "category": [u'Electronics', u'Computers & Accessories', u'Desktops'],\
        "aspect_candidate": ["drive","storage","keyboard","graphics","software","price","memory","monitor","processor","support"]
    },
    187:{
        "category": [u'Electronics', u'Computers & Accessories', u'Computer Components', u'Graphics Cards'],\
        "aspect_candidate": ["games","video","price","speed", "fan","drivers"]
    },
    402:{
        "category": [u'Electronics', u'Accessories & Supplies', u'Audio & Video Accessories', u'Headphones'],\
        "aspect_candidate":["sound","mic","microphone","comfortable","price","bass","cord"]
    },
    176:{
        "category":[u'Electronics', u'Computers & Accessories', u'Data Storage'],\
        "aspect_candidate": ["storage size","usb", "speed","install","performance","price","support","quiet"]
    },
    90:{
        "category":[u'Electronics', u'Home Audio', u'Stereo Components', u'Speakers'],\
        "aspect_candidate":["sound","bass","surround","price","setup","size"]
    },
    253:{
        "category":[u'Electronics', u'Car & Vehicle Electronics', u'Car Electronics', u'Car Audio'],\
        "aspect_candidate":["sound","radio","bass","price","quality","install","bluetooth","power","control"]
    }
}

changeAspectNameinDB(40)

category_id = 308 #Digital Cameras
sim_slope = 1  
sim_intercept = 0.2
seed_word_list = ["battery","pictures","price","zoom","easy","detection","design","video","quality","screen","size"]
wordlist_dict = get_wordlist_dict(category_id, seed_word_list, 10, sim_slope, sim_intercept, isPrint = False)
wordlist_dict = prune_wordlist_dict(wordlist_dict, isPrint = True)
writeWordlistDictToDB(category_id, wordlist_dict, rewrite_wordlist_dict_list = True)
for i in range(4):
    print
    print "{0} th iteration:".format(i + 2)
    wordlist_dict = get_wordlist_dict(category_id, wordlist_dict, 10, sim_slope, sim_intercept, predict_threshold = 1.05, isPrint = False)
    wordlist_dict = prune_wordlist_dict(wordlist_dict, isPrint = True)
    writeWordlistDictToDB(category_id, wordlist_dict, rewrite_wordlist_dict_list = False)

wordlist_dict = get_category_data_from_db(308, "wordlist_dict")

test_prediction(308,wordlist_dict, aspect_to_show = [], predict_threshold = 1.05, num_show = 100, show_no_feature = False)

