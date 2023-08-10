import sys
sys.path.append('..')

from twords.twords import Twords 
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import pandas as pd
# this pandas line makes the dataframe display all text in a line; useful for seeing entire tweets
pd.set_option('display.max_colwidth', -1)

twit = Twords()
twit.data_path = "../data/java_collector/brexit"
twit.background_path = '../jar_files_and_background/freq_table_72319443_total_words_twitter_corpus.csv'
twit.create_Background_dict()
twit.set_Search_terms(["brexit"])
twit.create_Stop_words()

twit.get_java_tweets_from_csv_list()

# find how many tweets we have in original dataset
print "Total number of tweets:", len(twit.tweets_df)

twit.keep_column_of_original_tweets()

twit.lower_tweets()

twit.keep_only_unicode_tweet_text()

twit.remove_urls_from_tweets()

twit.remove_punctuation_from_tweets()

twit.drop_non_ascii_characters_from_tweets()

twit.drop_duplicate_tweets()

twit.drop_by_search_in_name()

twit.convert_tweet_dates_to_standard()

twit.sort_tweets_by_date()

len(twit.tweets_df)

twit.keep_tweets_with_terms("brexit")

len(twit.tweets_df)

twit.create_word_bag()
twit.make_nltk_object_from_word_bag()
twit.create_word_freq_df(1000)

twit.word_freq_df.sort_values("log relative frequency", ascending = False, inplace = True)
twit.word_freq_df.head(20)

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
background_cutoff = 2000
twit.word_freq_df[twit.word_freq_df['background occurrences']>background_cutoff].sort_values("log relative frequency", ascending=True).set_index("word")["log relative frequency"][-num_words_to_plot:].plot.barh(figsize=(20,
                num_words_to_plot/2.), fontsize=30, color="c"); 
plt.title("log relative frequency", fontsize=30); 
ax = plt.axes();        
ax.xaxis.grid(linewidth=4);



