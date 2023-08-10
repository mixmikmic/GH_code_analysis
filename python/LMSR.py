import pandas as pd, numpy as np, pickle, ast

en_word2cluster = pickle.load(open("datasets/en_word2cluster.pickle", "rb"))
tr_word2cluster = pickle.load(open("datasets/tr_word2cluster.pickle", "rb"))

# en_word2cluster = dict()
# for w, vec in en_word2cluster_.items():
# #     en_word2cluster[w.decode("utf8")] = vec
#     en_word2cluster[str(w)] = vec

df = pd.read_csv("datasets/tokenized_reviews.csv")
df.head(5)

len(df[df.Language=="tr"])

def rev2vec(rev, clusters=False, clusters_dict_name=None, vectors_dict_name="en_vects",strategy="avg", vec_dim=300):
    strategies = {"avg":np.mean, "median":np.median, "sum":np.sum}
    if clusters:
        return strategies[strategy](np.array([globals()[clusters_dict_name][w] for w in rev.split(" ") if w in globals()[clusters_dict_name]]), axis=0)
    return strategies[strategy](np.array([globals()[vectors_dict_name][w] for w in rev.split(" ") if w in globals()[vectors_dict_name]]), axis=0)

def getvec(x, clusters=False):
    lang, rev = x.split(":::::")
    if clusters:
        return rev2vec(rev, clusters=True, clusters_dict_name=lang+"_word2cluster")
    return rev2vec(rev, lang+"_vects")

df_vectorized = df.copy()
df_vectorized["lang_rev"] = df_vectorized[["Language", "Review"]].apply(lambda x: x[0]+":::::"+x[1], axis=1)
df_vectorized["rev_vec"] = df_vectorized["lang_rev"].apply(lambda x:getvec(x, True))
df_vectorized = df_vectorized.drop(["lang_rev", "Review","tokenized_reviews"], axis=1)
df_vectorized.head(5)

df_vectorized.to_csv("datasets/LMSR_rev2vec.csv", index=False)



