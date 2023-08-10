import pandas as pd

ls

pwd

stop_word_df = pd.read_csv('/home/jovyan/capstone-52/Stopword_lists/AR_stopwords.csv', header=None)

stop_word_df.head()

stop_word_df.tail()

stop_word_df.columns = ["stop_words"]


dfList = stop_word_df['stop_words'].tolist()

dfList.to_pickle('/home/jovyan/capstone-52/topic_modeling_experiments/pickled_stopwords/comp_study_stopwords.p')

word_list = []
for i, word in stop_word_df.iterrows():
    word_list.append(word)

