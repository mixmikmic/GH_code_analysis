import Load_Text_Set as l_data
import run_Word2Vec as w2v

words = l_data.text_8(200000)
embeddings = w2v.run_embeddings()

import numpy as np
import regex as re

joy_words = ['happy','joy','pleasure','glee']
sad_words = ['sad','unhappy','gloomy']
scary_words = ['scary','frightening','terrifying', 'horrifying']
disgust_words = ['disgust', 'distaste', 'repulsion']
anger_words = ['anger','rage','irritated']

def syn_average(word, list_words = []):
    to_ret = 0
    count = 0 #use this in case a word isnt in dict
    for syn in list_words:
        if syn in words.dictionary:
            syn_id = words.dictionary[syn]
            to_ret+=np.matmul(embeddings[word].reshape(1,128), embeddings[syn_id].reshape(128,1))
            count +=1
        else:
            print(syn," is not in dict")
    return to_ret/count

def test(string_words):
    happy = words.dictionary['joy']
    sad = words.dictionary['fear']
    scary = words.dictionary['sad']
    disgust = words.dictionary['disgust']
    anger = words.dictionary['anger']
    
    
    d2happy = 0 
    d2sad = 0 
    d2scary = 0 
    d2disgust = 0
    d2anger = 0
    for a in string_words:
        if a in words.dictionary:
            in_dict = words.dictionary[a]
            d2happy += syn_average(in_dict,joy_words)
            d2sad += syn_average(in_dict,sad_words)
            d2scary += syn_average(in_dict,scary_words)
            d2disgust += syn_average(in_dict,disgust_words)
            d2anger += syn_average(in_dict,anger_words )
            
    d2happy = d2happy/len(string_words)
    d2sad = d2sad/len(string_words)
    d2scary = d2scary/len(string_words)
    d2disgust = d2disgust/len(string_words)
    d2anger = d2anger/len(string_words)
    print(  max(d2happy,0),"\t",max(d2sad,0),"\t", max(d2scary,0),"\t", max(d2disgust,0),"\t", max(d2anger,0))

def plot_emotions(top = 8):
    emotions= [ words.dictionary['joy'], words.dictionary['fear'],
        words.dictionary['sad'], words.dictionary['disgust'], words.dictionary['anger'] ]
        
    for i,i_word in enumerate(emotions):
        sim = embeddings.similarity(embeddings)        
        nearest = (-sim[i_word, :]).argsort()[1:top+1]
        print('Nearest to ', emotions[i], ": ")
        for k in range(top):
            close_word = words.reverse_dictionary(nearest[k])
            print('\t',close_word)
        
        
    

happy_string_ = "Even Harry, who knew nothing about the different brooms, thought it looked wonderful. Sleek and shiny, with a mahogany handle, it had a long tail of neat, straight twigs and Nimbus Two Thousand written in gold near the top. As seven o'clock drew nearer, Harry left the castle and set off in the dusk toward the Quidditch field. Held never been inside the stadium before. Hundreds of seats were raised in stands around the field so that the spectators were high enough to see what was going on. At either end of the field were three golden poles with hoops on the end. They reminded Harry of the little plastic sticks Muggle children blew bubbles through, except that they were fifty feet high. Too eager to fly again to wait for Wood, Harry mounted his broomstick and kicked off from the ground. What a feeling -- he swooped in and out of the goal posts and then sped up and down the field. The Nimbus Two Thousand turned wherever he wanted at his lightest touch."
scary_string = "and the next second, Harry felt Quirrell's hand close on his wrist. At once, a needle-sharp pain seared across Harry's scar; his head felt as though it was about to split in two; he yelled, struggling with all his might, and to his surprise, Quirrell let go of him. The pain in his head lessened -- he looked around wildly to see where Quirrell had gone, and saw him hunched in pain, looking at his fingers -- they were blistering before his eyes."
angry_string = 'He’d forgotten all about the people in cloaks until he passed a group of them next to the baker’s. He eyed them angrily as he passed. He didn’t know why, but they made him uneasy. This bunch were whispering  excitedly, too, and he couldn’t see a single collectingtin. It was on his way back past them, clutching a large doughnut in a bag, that he caught a few words of what they were saying.'

happy_string_words = re.sub(r"\p{P}+", "", happy_string_).split()
scary_string_words = re.sub(r"\p{P}+", "", scary_string).split()
angry_string_words = re.sub(r"\p{P}+", "",angry_string).split()
print("\n")
print("Sentence: ")
print(happy_string_)
print("Similarity to: ")
print("happy \t\t sad \t\t scary \t\t disgust \t\t anger")
test(happy_string_words)
print("\n")
print("Sentence: ")
print(scary_string)
print("Similarity to: ")
print("happy \t\t sad \t\t scary \t\t disgust \t\t anger")
test(scary_string_words)
print("\n")
print("Sentence: ")
print(angry_string)
print("Similarity to: ")
print("happy \t\t sad \t\t scary \t\t disgust \t\t anger")
test(angry_string_words)



