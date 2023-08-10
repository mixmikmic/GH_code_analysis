import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

get_ipython().magic('matplotlib inline')

df_bow = pd.read_csv("../reports/f1_score_bow_diff_train_size.csv")
df_word_embedd = pd.read_csv("../reports/f1_score_word_embedd_diff_train_size.csv")

traditional_methods = ["Grad Boost", "Grad Boost + Tfidf",
                       "Logistic Regression", "Logistic Regression + Tfidf",
                       "SVC", "SVC + Tfidf", "Multi NB", "Multi NB + Tfidf"]

word_embeddings = ["Doc2Vec", "Word2Vec", "Word2Vec + Tfidf", "GloVe", "GloVe + Tfidf"]

colors = ['r', 'b', '#f47be9', 'm', 'c', '#feff60', '#007485', '#7663b0']

X = np.array(df_bow["train_size"].tolist()) / 1000
y_bow = df_bow.drop(['train_size'], axis=1).values.tolist()
y_word_embedd = df_word_embedd.drop(['train_size'], axis=1).values.tolist()

ind_width = 10 / 32
x_ticks = np.arange(-5, 5, ind_width)
args = list(zip(*[x_ticks[i::8] for i in range(8)])) 

plt.figure(figsize=(25, 20))
plt.plot(X, y_bow, marker="o")
plt.xticks(X, X.astype(np.uint))

plt.legend(traditional_methods)
plt.ylabel("Accuracy")
plt.xlabel("Train Samples x1000")

plt.show()

plt.figure(figsize=(20, 18))
plt.plot(X, y_word_embedd, marker="o")
plt.xticks(X, X.astype(np.uint))

plt.legend(word_embeddings)
plt.ylabel("Accuracy")
plt.xlabel("Train Samples x1000")

plt.show()

plt.subplots(figsize=(20,15))

for i in range(0, len(X)):
    for j in range(0, len(y_bow[0])):
        plt.bar(i + args[i][j], y_bow[i][j], color=colors[j], label=traditional_methods[j], width=ind_width)

plt.ylabel("Accuracy")
plt.xlabel("Train Samples x1000")
plt.title("Bag Of Words")
plt.xticks([-4, -0.5, 3, 6.5], [50, 100, 150, 200])
plt.legend(traditional_methods)

plt.subplots(figsize=(20,15))

for i in range(0, len(X)):
    for j in range(0, len(y_word_embedd[0])):
        plt.bar(i + args[i][j], y_word_embedd[i][j], color=colors[j], label=word_embeddings[j], width=ind_width)

plt.ylabel("Accuracy")
plt.xlabel("Train Samples x1000")
plt.title("Word Embeddings + Logistic Regression")
plt.xticks([-4.4, -0.9, 2.65, 6.1], [50, 100, 150, 200])
plt.legend(word_embeddings, loc='upper left')

