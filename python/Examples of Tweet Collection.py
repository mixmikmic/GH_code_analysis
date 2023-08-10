import sys
sys.path.append('..')

from twords.twords import Twords 
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import pandas as pd
# this pandas line makes the dataframe display all text in a line; useful for seeing entire tweets
pd.set_option('display.max_colwidth', -1)

twit_mars = Twords()
# set path to folder that contains jar files for twitter search
twit_mars.jar_folder_path = "../jar_files_and_background/"

twit_mars.create_java_tweets(total_num_tweets=100, tweets_per_run=50, querysearch="mars rover",
                           final_until=None, output_folder="mars_rover",
                           decay_factor=4, all_tweets=True)

twit_mars.get_java_tweets_from_csv_list()

twit_mars.tweets_df.head(5)

twit = Twords()
twit.jar_folder_path = "../jar_files_and_background/"
twit.get_all_user_tweets("barackobama", tweets_per_run=500)

twit = Twords()
twit.data_path = "barackobama"
twit.get_java_tweets_from_csv_list()
twit.convert_tweet_dates_to_standard()

twit.tweets_df["retweets"] = twit.tweets_df["retweets"].map(int)
twit.tweets_df["favorites"] = twit.tweets_df["favorites"].map(int)

twit.tweets_df.sort_values("favorites", ascending=False)[:5]

twit.tweets_df.sort_values("retweets", ascending=False)[:5]

twit.background_path = '../jar_files_and_background/freq_table_72319443_total_words_twitter_corpus.csv'
twit.create_Background_dict()
twit.create_Stop_words()

twit.keep_column_of_original_tweets()
twit.lower_tweets()
twit.keep_only_unicode_tweet_text()
twit.remove_urls_from_tweets()
twit.remove_punctuation_from_tweets()
twit.drop_non_ascii_characters_from_tweets()
twit.drop_duplicate_tweets()
twit.convert_tweet_dates_to_standard()
twit.sort_tweets_by_date()

twit.create_word_bag()
twit.make_nltk_object_from_word_bag()
twit.create_word_freq_df(10000)

twit.word_freq_df.sort_values("log relative frequency", ascending = False, inplace = True)
twit.word_freq_df.head(20)

twit.tweets_containing("sotu")[:10]

num_words_to_plot = 32
background_cutoff = 100
twit.word_freq_df[twit.word_freq_df['background occurrences']>background_cutoff].sort_values("log relative frequency", ascending=True).set_index("word")["log relative frequency"][-num_words_to_plot:].plot.barh(figsize=(20,
                num_words_to_plot/2.), fontsize=30, color="c"); 
plt.title("log relative frequency", fontsize=30); 
ax = plt.axes();        
ax.xaxis.grid(linewidth=4);

num_words_to_plot = 50
background_cutoff = 1000
twit.word_freq_df[twit.word_freq_df['background occurrences']>background_cutoff].sort_values("log relative frequency", ascending=True).set_index("word")["log relative frequency"][-num_words_to_plot:].plot.barh(figsize=(20,
                num_words_to_plot/2.), fontsize=30, color="c"); 
plt.title("log relative frequency", fontsize=30); 
ax = plt.axes();        
ax.xaxis.grid(linewidth=4);

num_words_to_plot = 32
background_cutoff = 5000
twit.word_freq_df[twit.word_freq_df['background occurrences']>background_cutoff].sort_values("log relative frequency", ascending=True).set_index("word")["log relative frequency"][-num_words_to_plot:].plot.barh(figsize=(20,
                num_words_to_plot/2.), fontsize=30, color="c"); 
plt.title("log relative frequency", fontsize=30); 
ax = plt.axes();        
ax.xaxis.grid(linewidth=4);

num_words_to_plot = 32
background_cutoff = 5000
twit.word_freq_df[twit.word_freq_df['background occurrences']>background_cutoff].sort_values("log relative frequency", ascending=False).set_index("word")["log relative frequency"][-num_words_to_plot:].plot.barh(figsize=(20,
                num_words_to_plot/2.), fontsize=30, color="c"); 
plt.title("log relative frequency", fontsize=30); 
ax = plt.axes();        
ax.xaxis.grid(linewidth=4);



