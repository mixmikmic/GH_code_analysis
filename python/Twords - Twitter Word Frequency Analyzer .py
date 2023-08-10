# to reload files that are changed automatically
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

from twords.twords import Twords 
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
get_ipython().magic('matplotlib inline')
import pandas as pd
# this pandas line makes the dataframe display all text in a line; useful for seeing entire tweets
pd.set_option('display.max_colwidth', -1)

twit = Twords()
twit.data_path = "data/java_collector/charisma_300000"
twit.background_path = 'jar_files_and_background/freq_table_72319443_total_words_twitter_corpus.csv'
twit.create_Background_dict()
twit.set_Search_terms(["charisma"])
twit.create_Stop_words()

twit.get_java_tweets_from_csv_list()

# find how many tweets we have in original dataset
print "Total number of tweets:", len(twit.tweets_df)

twit.tweets_df.head(5)

# create a column that stores the raw original tweets
twit.keep_column_of_original_tweets()

twit.lower_tweets()

# for this data set this drops about 200 tweets
twit.keep_only_unicode_tweet_text()

twit.remove_urls_from_tweets()

twit.remove_punctuation_from_tweets()

twit.drop_non_ascii_characters_from_tweets()

# these will also commonly be used for collected tweets - DROP DUPLICATES and DROP BY SEARCH IN NAME
twit.drop_duplicate_tweets()

twit.drop_by_search_in_name()

twit.convert_tweet_dates_to_standard()

twit.sort_tweets_by_date()

# apparently not all tweets contain the word "charisma" at this point, so do this
# cleaning has already dropped about half of the tweets we started with 
len(twit.tweets_df)

twit.keep_tweets_with_terms("charisma")

len(twit.tweets_df)

twit.create_word_bag()
twit.make_nltk_object_from_word_bag()

# this creates twit.word_freq_df, a dataframe that stores word frequency values
twit.create_word_freq_df(400)

twit.word_freq_df.sort_values("log relative frequency", ascending = False, inplace = True)

twit.word_freq_df.head(20)

num_words_to_plot = 32
twit.word_freq_df.set_index("word")["log relative frequency"][-num_words_to_plot:].plot.barh(figsize=(20,
                num_words_to_plot/2.), fontsize=30, color="c"); 
plt.title("log relative frequency", fontsize=30); 
ax = plt.axes();        
ax.xaxis.grid(linewidth=4);

twit.tweets_containing("uniqueness").head(12)

twit.tweets_containing(" iba ").head(12)

twit.tweets_containing("carpenter").head(12)

twit.drop_by_term_in_tweet([" iba ", "minho", "flaming", "carpenter"])

# recompute word bag for word statistics
twit.create_word_bag()
twit.make_nltk_object_from_word_bag()

# create word frequency dataframe
twit.create_word_freq_df(400)

# plot new results - again, only showing top 30 here
twit.word_freq_df.sort_values("log relative frequency", ascending = True, inplace = True)

num_words_to_plot = 32
twit.word_freq_df.set_index("word")["log relative frequency"][-num_words_to_plot:].plot.barh(figsize=(20,
                num_words_to_plot/2.), fontsize=30, color="c"); 
plt.title("log relative frequency", fontsize=30); 
ax = plt.axes();        
ax.xaxis.grid(linewidth=4);

twit.word_freq_df.sort_values("log relative frequency", ascending=False, inplace=True)
twit.word_freq_df.head(12)

twit.tweets_containing("skill").head(12)

twit.tweets_df['username'].value_counts().head(12)

twit.tweets_by("pcyhrr").head(12)

twit.tweets_by("av_momo").head(12)

twit.drop_by_username_with_n_tweets(1)

# now recompute word bag and word frequency objeect and look at results
twit.create_word_bag()
twit.make_nltk_object_from_word_bag()

twit.create_word_freq_df(400)

twit.word_freq_df.sort_values("log relative frequency", ascending = True, inplace = True)

num_words_to_plot = 32
twit.word_freq_df.set_index("word")["log relative frequency"][-num_words_to_plot:].plot.barh(figsize=(20,
                num_words_to_plot/2.), fontsize=30, color="c"); 
plt.title("log relative frequency", fontsize=30); 
ax = plt.axes();        
ax.xaxis.grid(linewidth=4);

twit.word_freq_df.sort_values("log relative frequency", ascending=False).head(10)

twit.create_word_freq_df(10000)

twit.word_freq_df[twit.word_freq_df['background occurrences']>100].sort_values("log relative frequency", ascending=False).head(10)

num_words_to_plot = 32
background_cutoff = 100
twit.word_freq_df[twit.word_freq_df['background occurrences']>background_cutoff].sort_values("log relative frequency", ascending=True).set_index("word")["log relative frequency"][-num_words_to_plot:].plot.barh(figsize=(20,
                num_words_to_plot/2.), fontsize=30, color="c"); 
plt.title("log relative frequency", fontsize=30); 
ax = plt.axes();        
ax.xaxis.grid(linewidth=4);

num_words_to_plot = 32
background_cutoff = 500
twit.word_freq_df[twit.word_freq_df['background occurrences']>background_cutoff].sort_values("log relative frequency", ascending=True).set_index("word")["log relative frequency"][-num_words_to_plot:].plot.barh(figsize=(20,
                num_words_to_plot/2.), fontsize=30, color="c"); 
plt.title("log relative frequency", fontsize=30); 
ax = plt.axes();        
ax.xaxis.grid(linewidth=4);

num_words_to_plot = 32
background_cutoff = 2000
twit.word_freq_df[twit.word_freq_df['background occurrences']>background_cutoff].sort_values("log relative frequency", ascending=True).set_index("word")["log relative frequency"][-num_words_to_plot:].plot.barh(figsize=(20,
                num_words_to_plot/2.), fontsize=30, color="c"); 
plt.title("log relative frequency", fontsize=30); 
ax = plt.axes();        
ax.xaxis.grid(linewidth=4);

num_words_to_plot = 32
background_cutoff = 10000
twit.word_freq_df[twit.word_freq_df['background occurrences']>background_cutoff].sort_values("log relative frequency", ascending=True).set_index("word")["log relative frequency"][-num_words_to_plot:].plot.barh(figsize=(20,
                num_words_to_plot/2.), fontsize=30, color="c"); 
plt.title("log relative frequency", fontsize=30); 
ax = plt.axes();        
ax.xaxis.grid(linewidth=4);

smaller_charisma_df = twit.tweets_df[0:100].copy()

from langdetect import detect
import time

# this can take a while
start_time = time.time()
smaller_charisma_df["lang"] = smaller_charisma_df["text"].map(detect)
print "Took", round((time.time() - start_time)/60., 2), "minutes to compute"
sec_per_tweet = (time.time() - start_time)/float(len(smaller_charisma_df))
print "Took on average", round(sec_per_tweet,2), "seconds per tweet"
print "Can classify", round(60./sec_per_tweet,2), "tweets by language per minute"

smaller_charisma_df = smaller_charisma_df[smaller_charisma_df['lang'] == 'en']

smaller_charisma_df.head(4)

len(smaller_charisma_df)



