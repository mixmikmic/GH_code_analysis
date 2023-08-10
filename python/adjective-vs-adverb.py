import time
import random
import json
import en_core_web_md
from spacy import displacy
from spacy.tokens import Doc
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support

# Read all adjectives and adverbs that are present in English Wiktionary
# https://en.wiktionary.org/wiki/Wiktionary:Main_Page

with open("data/adjectives.txt", "r") as f:
    ADJ = set(line.strip() for line in f.readlines())

with open("data/adverbs.txt", "r") as f:
    ADV = set(line.strip() for line in f.readlines())
    
print("Total number of adjectives:", len(ADJ))
print("Total number of adverbs:", len(ADV))

# Learn to transform adjectives to adverbs

def transform_adj_to_adv(adjective):
    """
    Convert an adjective to the corresponding adverb.
    :param adjective: string (adjective)
    :return: string (adverb) or None
    """

    # friendly
    if adjective.endswith("ly"):
        return None
    # hard
    if adjective in ADV:
        return adjective

    # exceptions
    elif adjective == "good":
        return "well"
    elif adjective in ["whole", "true"]:
        return adjective[:-1] + "ly"

    # responsible => responsibly
    elif adjective.endswith("le") and adjective != "sole":
        adverb = adjective[:-1] + "y"
    # angry => angrily
    elif adjective.endswith("y") and adjective != "shy":
        adverb = adjective[:-1] + "ily"
    # idiotic => idiotically
    elif adjective.endswith("ic"):
        adverb = adjective + "ally"
    # full => fully
    elif adjective.endswith("ll"):
        adverb = adjective + "y"
    # free => freely
    else:
        adverb = adjective + "ly"

    # check for validity
    return adverb if adverb in ADV else None

for i in range(20):
    word = random.sample(ADJ, 1)[0]
    print("{:18} => {}".format(word, transform_adj_to_adv(word)))

# Create dictionaries for adjective-adverb transformation

adj_to_adv, adv_to_adj = dict(), dict()

for adj in ADJ:
    adv = transform_adj_to_adv(adj)
    if adv and adv != adj:
        adj_to_adv[adj] = adv
        adv_to_adj[adv] = adj

print("Total number of adjectives:", len(ADJ))
print("Total number of adverbs:", len(ADV))
print("Number of adjectives that can be transformed to adverbs:",
      len(adj_to_adv))

# Load spaCy models

start = time.time()
nlp = en_core_web_md.load(disable=['ner'])
print("Models loaded in", round(time.time() - start), "seconds.")

# Parse sentences with adjective and adverb

# sentence = nlp("The soup smells good.")
# print("Parts of speech:")
# print(" ".join("{}_{}".format(token.text, token.tag_) for token in sentence))
# displacy.render(sentence, style='dep', options={"collapse_punct": False, "distance": 110}, jupyter=True)

# sentence = nlp("He smells the hot soup carefully.")
# print("Parts of speech:")
# print(" ".join("{}_{}".format(token.text, token.tag_) for token in sentence))
# displacy.render(sentence, style='dep', options={"collapse_punct": False, "distance": 110}, jupyter=True)

# sentence = nlp("Mary naturally and quickly became part of our family.")
# print("Parts of speech:")
# print(" ".join("{}_{}".format(token.text, token.tag_) for token in sentence))
# displacy.render(sentence, style='dep', options={"collapse_punct": False, "distance": 110}, jupyter=True)

sentence = nlp("She was completely natural and unaffected by the attention.")
print("Parts of speech:")
print(" ".join("{}_{}".format(token.text, token.tag_) for token in sentence))
displacy.render(sentence, style='dep', options={"collapse_punct": False, "distance": 110}, jupyter=True)

# Collect features

def feature_extractor(sentence, ind):
    """
    Collect features for the INDth token in SENTENCE.
    
    :param sentence: Doc, a parsed sentence
    :param ind: the index of the token
    :return: a feature dictionary
    """
    token = sentence[ind]
    features = dict()
    # context
    features["w-1"] = sentence[ind-1].text if ind > 0 else "NONE"
    features["w+1"] = sentence[ind+1].text if ind < (len(sentence) - 1) else "NONE"
    # children
    for child in token.children:
        features[child.dep_] = child.text
    # if we collect features for an adjective
    if token.tag_ == "JJ" and token.text in adj_to_adv:
        features["adj"] = token.text
        features["adv"] = adj_to_adv[token.text]
        features["adj_head"] = token.dep_ + "_" + token.head.lemma_
        alt_sentence = nlp(" ".join([t.text for t in sentence[:ind]]
                                    + [features["adv"]] +
                                    [t.text for t in sentence[ind + 1:]]))
        features["adv_head"] = alt_sentence[ind].dep_ + "_" +                                alt_sentence[ind].head.lemma_
    # if we collect features for an adverb
    elif token.tag_ == "RB" and token.text in adv_to_adj:
        features["adv"] = token.text
        features["adj"] = adv_to_adj[token.text]
        features["adv_head"] = token.dep_ + "_" + token.head.lemma_
        alt_sentence = nlp(" ".join([t.text for t in sentence[:ind]]
                                    + [features["adj"]] +
                                    [t.text for t in sentence[ind + 1:]]))
        features["adj_head"] = alt_sentence[ind].dep_ + "_" +                                alt_sentence[ind].head.lemma_
    else:
        return None
    return features

# Collect features for sample sentences

corpus = ["The soup smells good.",
          "He smells the hot soup carefully.",
          "Mary naturally and quickly became part of our family.",
          "She was completely natural and unaffected by the attention."]
data, labels = [], []
for sentence in corpus:
    sentence = nlp(sentence)
    for token in sentence:
        if token.tag_ in ["JJ", "RB"] and token.head.tag_.startswith("VB"):
            features = feature_extractor(sentence, token.i)
            data.append(features)
            labels.append(token.pos)
            print("Word in question:", token.text)
            print(features)
            print("Label:", token.pos_)
            print("")

# Vectorize features for sample sentences

vec = DictVectorizer()
x = vec.fit_transform(data)

# The full feature set

print("All features:")
print(vec.get_feature_names())
print("\nTotal number of features: ", len(vec.get_feature_names()))

# The resulting sparse matrix

print("The resulting sparse matrix:")
print(x.toarray())

with open("data/adj_vs_adv_data.json", "r") as f:
    data = json.load(f)

for k, v in data[100].items():
    print(k + ":", v)

# Collect features from our data set

x_features, y = [], []
for sample in data:
    sentence = nlp(" ".join(sample["sentence"]))
    features = feature_extractor(sentence, sample["ind"])
    if features:
        x_features.append(features)
        y.append(sample["label"])

print(len(x_features), len(y))

# Vectorize data

print(x_features[100], y[100])

vectorizer = DictVectorizer()
x = vectorizer.fit_transform(x_features)
print("\nTotal number of features: ", len(vectorizer.get_feature_names()))

# Split data

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                    random_state=42)

# Train a classifier

lrc = LogisticRegression(random_state=42)
lrc.fit(x_train, y_train)
predicted = lrc.predict(x_test)
prec, rec, fscore, sup = precision_recall_fscore_support(y_test, predicted, labels=['ADJ', 'ADV'])
print("Precision:", [round(p, 2) for p in prec])
print("Recall:", [round(r, 2) for r in rec])
print("F-score:", [round(f, 2) for f in fscore])

def is_adj_correct(raw_sentence):
    sentence = nlp(raw_sentence)
    print("Input:", raw_sentence)
    for ind in range(len(sentence)):
        token = sentence[ind]
        if token.tag_ == "JJ" and token.head.tag_.startswith("VB"):
            features = feature_extractor(sentence, ind)
            predicted_pos = lrc.predict(vectorizer.transform(features))
            if predicted_pos == "ADJ":
                print("No errors found.")
                return
            else:
                print(" ".join([t.text for t in sentence[:ind]]
                                + ["{" + sentence[ind].text + "=>" + adj_to_adv[sentence[ind].text] + "}"] +
                                [t.text for t in sentence[ind+1:]]))
                return
    print("No errors found.")
    return

is_adj_correct("You have successful completed the project .")
print("")
is_adj_correct("I am busy talking to my friend.")
print("")
is_adj_correct("I am emotional talking to my friend.")
print("")
is_adj_correct("The soup smells good .")
print("")
is_adj_correct("He smells the hot soup careful .")
print("")
is_adj_correct("Mary natural and quickly became part of our family.")
print("")
is_adj_correct("She was completely natural and unaffected by the attention.")



