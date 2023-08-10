data=pd.read_pickle('pickels/16k_apperal_data')

stop_words=set(stopwords.words('english'))
print('list of stop words: \n',stop_words)

def nlp_preprocessing(total_text, index, column):
    if type(total_text) is not int:
        string = ""
        for word in total_text.split():
            word=("".join(e for e in word if e.isalnum()))
            word=word.lower()
            if word not in stop_words :
                string = string + word+" "
        data[column][index]= string

start_time=time.clock()
for index,row in data.iterrows():
    nlp_preprocessing(row['title'],index,'title')
print( time.clock() - start_time,'seconds' )

data.head()

data.to_pickle('pickels/16k_apperal_data_preprocessed')



