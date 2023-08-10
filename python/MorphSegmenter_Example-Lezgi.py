#API for parsing XML docs
import xml.etree.ElementTree as ET
from itertools import chain
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import sklearn
import pycrfsuite
from collections import Counter

def XMLtoWords(filename):
    '''Takes FLExText text as .xml. Returns data as list: [[[[[[morpheme, gloss], pos],...],words],sents]].
    Ignores punctuation. Morph_types can be: stem, suffix, prefix, or phrase when lexical item is made up of two words.'''
    
    datalists = []

    #open XML doc using xml parser
    root = ET.parse(filename).getroot()

    for text in root:
        for paragraphs in text:
            #Only get paragraphs, ignore metadata.
            if paragraphs.tag == 'paragraphs':
                for paragraph in paragraphs:
                    #jump straight into items under phrases
                    for phrase in paragraph[0]:
                        sent = []
                        #ignore first item which is the sentence number
                        for word in phrase[1]:
                            #ignore punctuation tags which have no attributes
                            if word.attrib:
                                lexeme = []
                                for node in word:
                                    if node.tag == 'morphemes':
                                        for morph in node:
                                            morpheme = []
                                            #note morph type 
                                            morph_type = morph.get('type')
                                            #Treat MWEs or unlabled morphemes as stems.
                                            if morph_type == None or morph_type == 'phrase':
                                                morph_type = 'stem'                                            
                                            for item in morph:
                                                #get morpheme token
                                                if item.get('type') == 'txt':
                                                    form = item.text
                                                    #get rid of hyphens demarcating affixes
                                                    if morph_type == 'suffix':
                                                        form = form[1:]
                                                    if morph_type == 'prefix':
                                                        form = form[:-1]
                                                    morpheme.append(form)
                                                #get affix glosses
                                                if item.get('type') == 'gls' and morph_type != 'stem':
                                                    morpheme.append(item.text)
                                            #get stem "gloss" = 'stem'
                                            if morph_type == 'stem':
                                                morpheme.append(morph_type)
                                            lexeme.append(morpheme)
                                    #get word's POS
                                    if node.get('type') == 'pos':
                                        lexeme.append(node.text)
                                sent.append(lexeme)
                        datalists.append(sent)
    return datalists

def WordsToLetter(wordlists):
    '''Takes data from XMLtoWords. Returns [[[[[letter, POS, BIO-label],...],words],sents]]'''

    letterlists = []
    
    for phrase in wordlists:
        sent = []
        for lexeme in phrase:
            word = []
            #Skip POS label
            for morpheme in lexeme[:-1]:
                #use gloss as BIO label
                label = morpheme[1]
                #Break morphemes into letters
                for i in range(len(morpheme[0])):
                    letter = [morpheme[0][i]]
                    #add POS label to each letter
                    letter.append(lexeme[-1])
                    #add BIO label
                    if i == 0:
                        letter.append('B-' + label)
                    else:
                        letter.append('I')
                    word.append(letter)
            sent.append(word)
        letterlists.append(sent)
    
    return letterlists

#Randomize and split the data
traindata,testdata = train_test_split(WordsToLetter(XMLtoWords("FLExTxtExport2.xml")),test_size=0.2)

def extractFeatures(sent):
    '''Returns letters of a sentence with features list.'''
    
    featurelist = []
    senlen = len(sent)
    
    #each word in a sentence
    for i in range(senlen):
        word = sent[i]
        wordlen = len(word)
        lettersequence = ''
        #each letter in a word
        for j in range(wordlen):
            letter = word[j][0]
            #gathering previous letters
            lettersequence += letter
            #ignore digits             
            if not letter.isdigit():
                features = [
                    'bias',
                    'letterLowercase=' + letter.lower(),
                    'postag=' + word[j][1],
                ] 
                #position of word in sentence and pos tags sequence
                if i > 0:
                    features.append('prevpostag=' + sent[i-1][0][1])
                    if i != senlen-1:
                        features.append('nxtpostag=' + sent[i+1][0][1])
                    else:
                        features.append('EOS')
                else:
                    features.append('BOS')
                    #Don't get pos tag if sentence is 1 word long
                    if i != senlen-1:
                        features.append('nxtpostag=' + sent[i+1][0][1])
                #position of letter in word
                if j == 0:
                    features.append('BOW')
                elif j == wordlen-1:
                    features.append('EOW')
                else:
                    features.append('letterposition=-%s' % str(wordlen-1-j))
                #letter sequences before letter
                if j >= 4:
                    features.append('prev4letters=' + lettersequence[j-4:j].lower() + '>')
                if j >= 3:
                    features.append('prev3letters=' + lettersequence[j-3:j].lower() + '>')
                if j >= 2:
                    features.append('prev2letters=' + lettersequence[j-2:j].lower() + '>')
                if j >= 1:
                    features.append('prevletter=' + lettersequence[j-1:j].lower() + '>')
                #letter sequences after letter
                if j <= wordlen-2:
                    nxtlets = word[j+1][0]
                    features.append('nxtletter=<' + nxtlets.lower())
                    #print('\nnextletter:', nxtlet)
                if j <= wordlen-3:
                    nxtlets += word[j+2][0]
                    features.append('nxt2letters=<' + nxtlets.lower())
                    #print('next2let:', nxt2let)
                if j <= wordlen-4:
                    nxtlets += word[j+3][0]
                    features.append('nxt3letters=<' + nxtlets.lower())
                if j <= wordlen-5:
                    nxtlets += word[j+4][0]
                    features.append('nxt4letters=<' + nxtlets.lower())
                
            featurelist.append(features)
    
    return featurelist

def extractLabels(sent):
    labels = []
    for word in sent:
        for letter in word:
            labels.append(letter[2])
    return labels

def extractTokens(sent):
    tokens = []
    for word in sent:
        for letter in word:
            tokens.append(letter[0])
    return tokens

def sent2features(data):
    return [extractFeatures(sent) for sent in data]

def sent2labels(data):
    return [extractLabels(sent) for sent in data]

def sent2tokens(data):
    return [extractTokens(sent) for sent in data]

X_train = sent2features(traindata)
Y_train = sent2labels(traindata)

X_test = sent2features(testdata)
Y_test = sent2labels(testdata)

trainer = pycrfsuite.Trainer(verbose=False)

for xseq, yseq in zip(X_train, Y_train):
    trainer.append(xseq, yseq)

trainer.set_params({
        'c1': 1.0, #coefficient for L1 penalty
        'c2': 1e-3, #coefficient for L2 penalty
        'max_iterations': 50 #early stopping
    })

model_filename = 'LING5800_lezgi.crfsuite'
trainer.train(model_filename)

tagger = pycrfsuite.Tagger()
tagger.open(model_filename)

example_sent = testdata[0]
print('Letters:', '  '.join(extractTokens(example_sent)), end='\n')

print('Predicted:', ' '.join(tagger.tag(extractFeatures(example_sent))))
print('Correct:', ' '.join(extractLabels(example_sent)))

def bio_classification_report(y_correct, y_pred):
    '''Takes list of correct and predicted labels from tagger.tag. 
    Prints a classification report for a list of BIO-encoded sequences.
    It computes letter-level metrics.'''

    labeler = LabelBinarizer()
    y_correct_combined = labeler.fit_transform(list(chain.from_iterable(y_correct)))
    y_pred_combined = labeler.transform(list(chain.from_iterable(y_pred)))
    
    tagset = set(labeler.classes_)
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(labeler.classes_)}
    
    return classification_report(
        y_correct_combined,
        y_pred_combined,
        labels = [class_indices[cls] for cls in tagset],
        target_names = tagset)

Y_pred = [tagger.tag(xseq) for xseq in X_test]

print(bio_classification_report(Y_test, Y_pred))

info = tagger.info()

def print_transitions(trans_features):
    '''Print info from the crfsuite.'''
    
    for (label_from, label_to), weight in trans_features:
        print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))

print("Top likely transitions:")
print_transitions(Counter(info.transitions).most_common(15))

print("\nTop unlikely transitions:")
print_transitions(Counter(info.transitions).most_common()[-15:])

def print_state_features(state_features):
    for (attr, label), weight in state_features:
        print("%0.6f %-6s %s" % (weight, label, attr))    

print("Top positive:")
print_state_features(Counter(info.state_features).most_common(15))

print("\nTop negative:")
print_state_features(Counter(info.state_features).most_common()[-15:])

