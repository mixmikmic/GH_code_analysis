import pandas as pd, numpy as np, ast, re, pickle, ast
np.random.seed(42)

def parse_np_array(array_string, as_nparray=True):
    pattern = r'''# Match (mandatory) whitespace between...
              (?<=\]) # ] and
              \s+
              (?= \[) # [, or
              |
              (?<=[^\[\]\s]) 
              \s+
              (?= [^\[\]\s]) # two non-bracket non-whitespace characters
           '''
    fixed_string = re.sub(pattern, ',', array_string, flags=re.VERBOSE)
    if as_nparray:
        return np.array(ast.literal_eval(fixed_string))
    return ast.literal_eval(fixed_string)

df = pd.read_csv("datasets/LMS_r_merged_reviews_per_movie_language_score.csv")
df["merged_reviews_vector"] = df["merged_reviews_vector"].apply(lambda x: parse_np_array(x) if type(x) == str and "[" in x else None)
df.head(5)

df.groupby(["Language","Score"]).count()

def merging_function(frame):
    return np.mean(frame["merged_reviews_vector"])

merged_by_lang_and_movies = df.groupby(["Language","Score"], as_index=False).apply(merging_function).to_frame()
merged_by_lang_and_movies

merged_by_lang_and_movies.reset_index(inplace=True)

"There are {} movies".format(len(df.groupby("Movie_ID")))

def mikolov(X, Y, W):
    # min_W  for each i    ||W.x(i) - y(i)||^2
    result = 0
    for score in range(len(X)):
        result += np.linalg.norm(W.dot(X[score]) - Y[score])**2
    return result

en_revs = dict()
tr_revs = dict()
for movie in df.set_index("Movie_ID").iterrows():
    vec = movie[1]["merged_reviews_vector"]
    lang = movie[1]["Language"]
    score = movie[1]["Score"]
    if lang == "en":
        en_revs[score] = vec
    else:
        tr_revs[score] = vec

def learn_translation_matrix(X,Y, iterations=5000, alpha=0.0001, alpha_change_rate=0.8):
    W = np.random.random((300, 300))
    for i in range(iterations+1):
        gradient = np.zeros(300)
        for score in range(len(X)):
            error = X[score].dot(W) - Y[score]
            gradient += alpha * np.gradient(error)
        W += gradient
        if i == 2000:
            alpha /= 100

        if i%1000 == 0:
            alpha *= alpha_change_rate
            print("Mikolov distance: {}".format(mikolov(X, Y, W)))
    return W

scores = sorted([i for i in tr_revs.keys() if i in en_revs.keys()])

En_score_vecs = np.array([en_revs[sv] for sv in scores])  # English score vectors
Tr_score_vecs = np.array([tr_revs[sv] for sv in scores])  # Turkish score vectors

from sklearn.neural_network import MLPRegressor

W = MLPRegressor()
W.fit(En_score_vecs, Tr_score_vecs)

# def merge_cross_lingual_score_vectors(En_score_vecs, Tr_score_vecs, scores, W):
#     labeled_vecs = dict()
#     for score in range(len(scores)):
#         labeled_vecs[scores[score]] = np.mean(\
#             np.array(\
#                 [En_score_vecs[score].dot(W), Tr_score_vecs[score]]\
#                     ), axis=0)
#     return labeled_vecs
def merge_cross_lingual_score_vectors(En_score_vecs, Tr_score_vecs, scores, W):
    labeled_vecs = dict()
    for score in range(len(scores)):
        labeled_vecs[scores[score]] = np.mean(            W.predict(np.atleast_2d(En_score_vecs[score])                    ), axis=0)
    return labeled_vecs

labeled_vecs = merge_cross_lingual_score_vectors(En_score_vecs, Tr_score_vecs, scores, W)

pickle.dump(labeled_vecs, open("score_vectors_dict", "wb"))



