import pandas as pd
import re
pd.set_option('display.max_colwidth',200)

df = pd.read_csv(r'/Users/arm/Downloads/chatbot_training_trump.csv')
df

df.shape

conversation = df.iloc[:,0]
conversation

clist = []
def question_answer_pairs(x): 
    conversation_pairs = re.findall(r": (.*?)(?:$|\\n)", x)
    clist.extend(list(zip(conversation_pairs, conversation_pairs[0:])))
conversation.map(question_answer_pairs);
conversation_frame = pd.Series(dict(clist)).to_frame().reset_index()
conversation_frame.columns = ['question', 'answer']

conversation_frame.head()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
vectorizer = TfidfVectorizer(ngram_range=(1,3))
vec = vectorizer.fit_transform(conversation_frame['question'])

my_q = vectorizer.transform(['Hi. What is your name?'])
cs = cosine_similarity(my_q, vec)
rs = pd.Series(cs[0]).sort_values(ascending=False)
top5 = rs.iloc[0:5]
top5

conversation_frame.iloc[top5.index]['question']

rsi = rs.index[0]
rsi
conversation_frame.iloc[rsi]['answer']

def get_response(question):
    my_question = vectorizer.transform([question])
    cs = cosine_similarity(my_question, vec)
    rs = pd.Series(cs[0]).sort_values(ascending=False)
    rsi = rs.index[0]
    return conversation_frame.iloc[rsi]['answer']

get_response("I'm smarter than you'll ever be")

get_response('Who do you think you are?')

get_response("My favorite color is blue. What's yours")

get_response("Do you have any questions for me?")

get_response("What do you do for fun?")

get_response("How do you think others would describe you?")

get_response("Is there anything you'd like to apologize for?")

get_response("Do you have any regrets?")

get_response("What do you do to relax?")

get_response("Say goodbye, Trump")

get_response("What advise would you give yourself in the future?")

