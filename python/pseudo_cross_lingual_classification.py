import pandas as pd, numpy as np, ast, re
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

def parse_np_array(array_string):
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
    return np.array(ast.literal_eval(fixed_string))

df = pd.read_csv("labeled_bow.csv", index_col="ID")
df["bow_word2vec"] = df["bow_word2vec"].apply(lambda x: parse_np_array(x) if type(x) == str and "[" in x else None)
df["bow_clust2vec"] = df["bow_clust2vec"].apply(lambda x: parse_np_array(x) if type(x) == str and "[" in x else None)
df.head(5)

y = np.array(df["Score"])
X_w2v = np.array(list(df["bow_word2vec"]))
X_c2v = np.array(list(df["bow_clust2vec"]))

ss = StandardScaler()
X_w2v = ss.fit_transform(X_w2v)
X_c2v = ss.fit_transform(X_c2v)

wX_train, wX_test, y_train, y_test = train_test_split(X_w2v, y, test_size=0.30, random_state=42)
cX_train, cX_test, y_train, y_test = train_test_split(X_c2v, y, test_size=0.30, random_state=42)
wX_train.shape, wX_test.shape

def eval_model(model, X_train, X_test, y_train, y_test, fit=True):
    if fit:
        model.fit(X_train, y_train)
    predicted_y = model.predict(X_test)
    return 1 - np.sum(np.abs(y_test - predicted_y))/(len(y_test)*10)

sgd_clf = SGDClassifier(random_state=42, max_iter=200, alpha=0.01, )
rf_clf = RandomForestClassifier(random_state=42, n_estimators=5)
svc_clf = LinearSVC(random_state=42, multi_class="crammer_singer")
# lr_clf = LogisticRegression(random_state=42, C=0.5, max_iter=200, multi_class="multinomial")
mlp_clf = MLPClassifier(random_state=42, max_iter=500)

for model in ["sgd_clf", "rf_clf", "svc_clf", "mlp_clf"]:
    print(model)
    print("BOW_word2vec: ",eval_model(globals()[model], wX_train, wX_test, y_train, y_test))
    print("BOW_clust2vec: ",eval_model(globals()[model], cX_train, cX_test, y_train, y_test))

class CLassification:
    def __init__(self, df_path, only_tr=False):
        self.df = self.load_df(df_path, only_tr)
        
        self.X_w2v, self.X_c2v, self.y = self.load_Xy()
        
        self.wX_train, self.wX_test, self.y_train, self.y_test = train_test_split(            X_w2v, y, test_size=0.30, random_state=42)
        
        self.cX_train, self.cX_test, self.y_train, self.y_test = train_test_split(            X_c2v, y, test_size=0.30, random_state=42)
        
        
    def load_df(self, df_path, only_tr):
        df = pd.read_csv(df_path, index_col="ID")
        if only_tr:
            df = df[df["language"] == "tr"]
        df["bow_word2vec"] = df["bow_word2vec"].apply(lambda x: parse_np_array(x) if type(x) == str and "[" in x else None)
        df["bow_clust2vec"] = df["bow_clust2vec"].apply(lambda x: parse_np_array(x) if type(x) == str and "[" in x else None)
        return df

    def load_Xy(self):
        y = np.array(self.df["Score"])
        X_w2v = self.scale_X(np.array(list(self.df["bow_word2vec"])))
        X_c2v = self.scale_X(np.array(list(self.df["bow_clust2vec"])))
        return X_w2v, X_c2v, y
    
    def scale_X(self, X):
        ss = StandardScaler()
        X = ss.fit_transform(X)
        
    def eval_model(self, model, X_train, X_test, y_train, y_test, fit=True):
        if fit:
            model.fit(X_train, y_train)
        predicted_y = model.predict(X_test)
        return 1 - np.sum(np.abs(y_test - predicted_y))/(len(y_test)*10) 
    
    def eval_data(self):
        sgd_clf = SGDClassifier(random_state=42, max_iter=200, alpha=0.01, )
        rf_clf = RandomForestClassifier(random_state=42, n_estimators=5)
        svc_clf = LinearSVC(random_state=42, multi_class="crammer_singer")
        # lr_clf = LogisticRegression(random_state=42, C=0.5, max_iter=200, multi_class="multinomial")
        mlp_clf = MLPClassifier(random_state=42, max_iter=500)
        for model in ["sgd_clf", "rf_clf", "svc_clf", "mlp_clf"]:
            print(model)
            print("BOW_word2vec: ",eval_model(locals()[model], self.wX_train, self.wX_test, self.y_train, self.y_test))
            print("BOW_clust2vec: ",eval_model(locals()[model], self.cX_train, self.cX_test, self.y_train, self.y_test))

cl = CLassification("labeled_bow_pcl.csv")  ## pcl: pseudo-cross-lingual
cl.eval_data()

cl = CLassification("labeled_bow_pcl.csv", only_tr=True)  ## pcl: pseudo-cross-lingual
cl.eval_data()

cl = CLassification("labeled_bow.csv", only_tr=True)  ## pcl: pseudo-cross-lingual
cl.eval_data()



