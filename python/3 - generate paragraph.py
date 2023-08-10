from __future__ import print_function
import numpy as np
import os
import scipy
from six.moves import cPickle

save_dir = 'save' # directory to store models

#import spacy, and french model
import spacy
nlp = spacy.load('fr')

#import gensim library
import gensim
from gensim.models.doc2vec import LabeledSentence

#load the doc2vec model
print("loading doc2Vec model...")
d2v_model = gensim.models.doc2vec.Doc2Vec.load('./data/doc2vec.w2v')

print("model loaded!")

#load vocabulary
print("loading vocabulary...")
vocab_file = os.path.join(save_dir, "words_vocab.pkl")

with open(os.path.join(save_dir, 'words_vocab.pkl'), 'rb') as f:
        words, vocab, vocabulary_inv = cPickle.load(f)

vocab_size = len(words)
print("vocabulary loaded !")

from keras.models import load_model
# load the keras models
print("loading word prediction model...")
model = load_model(save_dir + "/" + 'my_model_gen_sentences_lstm.final.hdf5')
print("model loaded!")
print("loading sentence selection model...")
model_sequence = load_model(save_dir + "/" + 'my_model_sequence_lstm.final.hdf5')
print("model loaded!")

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def create_seed(seed_sentences,nb_words_in_seq=20, verbose=False):
    #initiate sentences
    generated = ''
    sentence = []
    
    #fill the sentence with a default word
    for i in range (nb_words_in_seq):
        sentence.append("le")

    seed = seed_sentences.split()
    
    if verbose == True : print("seed: ",seed)

    for i in range(len(sentence)):
        sentence[nb_words_in_seq-i-1]=seed[len(seed)-i-1]
        #print(i, sentence)

    generated += ' '.join(sentence)
    
    if verbose == True : print('Generating text with the following seed: "' + ' '.join(sentence) + '"')

    return [generated, sentence]

def generate_phrase(sentence, max_words = 50, nb_words_in_seq=20, temperature=1, verbose = False):
    generated = ""
    words_number = max_words - 1
    ponctuation = [".","?","!",":","…"]
    seq_length = nb_words_in_seq
    #sentence = []
    is_punct = False
    
    #generate the text
    for i in range(words_number):
        #create the vector
        x = np.zeros((1, seq_length, vocab_size))
        for t, word in enumerate(sentence):
            #print(t, word, vocab[word])
            x[0, nb_words_in_seq-len(sentence)+t, vocab[word]] = 1.
        #print(x.shape)

        #calculate next word
        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, temperature)
        next_word = vocabulary_inv[next_index]
        
        if verbose == True:
            predv = np.array(preds)
            #arr = np.array([1, 3, 2, 4, 5])
            wi = predv.argsort()[-3:][::-1]
            print("potential next words: ", vocabulary_inv[wi[0]], vocabulary_inv[wi[1]], vocabulary_inv[wi[2]])

        #add the next word to the text
        if is_punct == False:
            if next_word in ponctuation:
                is_punct = True
            generated += " " + next_word
            # shift the sentence by one, and and the next word at its end
            sentence = sentence[1:] + [next_word]

    return(generated, sentence)

def define_phrases_candidates(sentence, max_words = 50,                              nb_words_in_seq=20,                               temperature=1,                               nb_candidates_sents=10,                               verbose = False):
    phrase_candidate = []
    generated_sentence = ""
    for i in range(nb_candidates_sents):
        generated_sentence, new_sentence = generate_phrase(sentence,                                                            max_words = max_words,                                                            nb_words_in_seq = nb_words_in_seq,                                                            temperature=temperature,                                                            verbose = False)
        phrase_candidate.append([generated_sentence, new_sentence])
    
    if verbose == True :
        for phrase in phrase_candidate:
            print("   " , phrase[0])
    return phrase_candidate

def create_sentences(doc):
    ponctuation = [".","?","!",":","…"]
    sentences = []
    sent = []
    for word in doc:
        if word.text not in ponctuation:
            if word.text not in ("\n","\n\n",'\u2009','\xa0'):
                sent.append(word.text.lower())
        else:
            sent.append(word.text.lower())
            if len(sent) > 1:
                sentences.append(sent)
            sent=[]
    return sentences

def generate_training_vector(sentences_list, verbose = False):
    if verbose == True : print("generate vectors for each sentence...")
    seq = []
    V = []

    for s in sentences_list:
        #infer the vector of the sentence, from the doc2vec model
        v = d2v_model.infer_vector(create_sentences(nlp(s))[0], alpha=0.001, min_alpha=0.001, steps=10000)
    #create the vector array for the model
        V.append(v)
    V_val=np.array(V)
    #expand dimension to fit the entry of the model : that's the training vector
    V_val = np.expand_dims(V_val, axis=0)
    if verbose == True : print("Vectors generated!")
    return V_val

def select_next_phrase(model, V_val, candidate_list, verbose=False):
    sims_list = []
    
    #calculate prediction
    preds = model.predict(V_val, verbose=0)[0]
    
    #calculate vector for each candidate
    for candidate in candidate_list:
        #calculate vector
        #print("calculate vector for : ", candidate[1])
        V = np.array(d2v_model.infer_vector(candidate[1]))
        #calculate csonie similarity
        sim = scipy.spatial.distance.cosine(V,preds)
        #populate list of similarities
        sims_list.append(sim)
    
    #select index of the biggest similarity
    m = max(sims_list)
    index_max = sims_list.index(m)
    
    if verbose == True :
        print("selected phrase :")
        print("     ", candidate_list[index_max][0])
    return candidate_list[index_max]

def generate_paragraphe(phrase_seed, sentences_seed,                         max_words = 50,                         nb_words_in_seq=20,                         temperature=1,                         nb_phrases=30,                         nb_candidates_sents=10,                         verbose=True):
    
    sentences_list = sentences_seed
    sentence = phrase_seed   
    text = []
    
    for p in range(nb_phrases):
        if verbose == True : print("")
        if verbose == True : print("#############")
        print("phrase ",p+1, "/", nb_phrases)
        if verbose == True : print("#############")       
        if verbose == True:
            print('Sentence to generate phrase : ')
            print("     ", sentence)
            print("")
            print('List of sentences to constrain next phrase : ')
            print("     ", sentences_list)
            print("")
    
        #generate seed training vector
        V_val = generate_training_vector(sentences_list, verbose = verbose)

        #generate phrase candidate
        if verbose == True : print("generate phrases candidates...")
        phrases_candidates = define_phrases_candidates(sentence,                                                        max_words = max_words,                                                        nb_words_in_seq = nb_words_in_seq,                                                        temperature=temperature,                                                        nb_candidates_sents=nb_candidates_sents,                                                        verbose = verbose)
        
        if verbose == True : print("select next phrase...")
        next_phrase = select_next_phrase(model_sequence,                                          V_val,
                                         phrases_candidates, \
                                         verbose=verbose)
        
        print("Next phrase: ",next_phrase[0])
        if verbose == True :
            print("")
            print("Shift phrases in sentences list...")
        for i in range(len(sentences_list)-1):
            sentences_list[i]=sentences_list[i+1]

        sentences_list[len(sentences_list)-1] = next_phrase[0]
        
        if verbose == True:
            print("done.")
            print("new list of sentences :")
            print("     ", sentences_list)     
        sentence = next_phrase[1]
        
        text.append(next_phrase[0])
    
    return text

s1 = "nolan s' approche du bord du chemin et regarde en contrebas ."
s2 = "il se tourne vers mara :"
s3 = "- que dis tu ?"
s4 = "- rien du tout , lui répond la jeune femme en détournant le regard ."
s5 = "- je t' ai entendu dire quelque chose , pourtant ."
s6 = "- je pensais à voix haute , explique mara  ."
s7 = "l' apprentie hésite , elle n' est pas certaine que nolan comprenne ."
s8 = "depuis quelques jours , nolan est à fleur de peau et s'inquiète pour un rien ."
s9 = "- je crois avoir vu une ombre , déclare finalement la jeune femme ."
s10 = "- à quel endroit ?"
s11 = "s' écrie le jeune homme ."
s12 = "nolan semble bouleversé et il est devenu blanc de peur ."
s13 = "les souvenirs des kaurocs sont suffisament frais dans sa mémoire pour qu' une étrange angoisse lui noue la poitrine ."
s14 = "- ne sois pas inquiet , s' exclame mara , confuse de la réaction de son ami ."
s15 = "il y a probablement une erreur ."

sentences_list = [s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15]
print(sentences_list)

phrase_seed, sentences_seed = create_seed(s1 + " " + s2 + " " +                                          s3 + " " + s4+ " " + s5 + " " +                                          s6 + " " + s7 + " " + s8 + " " +                                          s9+ " " + s10 + " " + s11 + " " +                                          s12 + " " + s13 + " " + s14+ " " + s15,20)
print(phrase_seed)
print(sentences_seed)

text = generate_paragraphe(sentences_seed, sentences_list,                            max_words = 80,                            nb_words_in_seq = 30,                           temperature=0.201,                            nb_phrases=5,                            nb_candidates_sents=7,                            verbose=False)

print("generated text: ")
for t in text:
    print(t)



