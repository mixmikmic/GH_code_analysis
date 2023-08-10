from __private import fs

import pandas as pd

from os import listdir
from classification import dao

data_directory = "./example_data/"
twitter_data = [data_directory+fn for fn in listdir(data_directory)]

for _ in twitter_data: print(_)

for _ in fs.list(): print(_)

clf_fn_alc = "alcohol|accuracy:0.8143360752056404|f1:0.8192219679633866|type:LogisticRegression"
clf_fn_fpa = "first_person|accuracy:0.7112299465240641|f1:0.8021978021978021|type:SVC"
clf_fn_fpl = "first_person_label|accuracy:0.5637860082304527|f1:0.5643693591852614|type:LogisticRegression"

get_ipython().run_cell_magic('time', '', '\nclf_alc = dao.ClassifierAccess.get_byfile(clf_fn_alc)\nclf_fpa = dao.ClassifierAccess.get_byfile(clf_fn_fpa)\nclf_fpl = dao.ClassifierAccess.get_byfile(clf_fn_fpl)')

clf_fpl.get_params()

get_ipython().run_cell_magic('time', '', '\nfrom data import DataAccess, LabelGetter\n\nX = DataAccess.get_as_dataframe()\nL = LabelGetter(X)')

get_ipython().run_cell_magic('time', '', 'clf_fpl.fit(*L.get_first_person_label())')

get_ipython().run_cell_magic('time', '', 'clf_fpa.fit(*L.get_first_person())')

df = pd.read_csv(twitter_data[1])
print(len(df))

df.head()

predictions_alc = clf_alc.predict_proba(df)

print(predictions_alc.shape)
predictions_alc

df["predict_alc"] = predictions_alc[:,1]

df.head()

thres = 0.75
filter_alc = df.predict_alc > thres

df["predict_fpa|alc"] = 0 

predict_fpa = clf_fpa.predict_proba(df[filter_alc])
df.loc[filter_alc, "predict_fpa|alc"] = predict_fpa[:,1]

df["predict_fpa"] = df["predict_alc"] * df["predict_fpa|alc"]

predict_fpl = clf_fpl.predict_proba(df[filter_alc])

predict_fpl = pd.DataFrame(
    predict_fpl, 
         columns=[
        "predict_present|fpa", 
        "predict_future|fpa", 
        "predict_past|fpa"],
    index=df[filter_alc].index)

for col in predict_fpl.columns:
    predict_fpl[col.split("|")[0]] = predict_fpl[col] * df[filter_alc]["predict_fpa"]

df = df.join(predict_fpl).fillna(0)

df[df.predict_alc > thres].head()

df.columns

class PredictionTransformer:
    
    cols = [
        'predict_alc', 
        'predict_fpa', 
        'predict_fpa|alc',
    ]
    
    def __init__(self, clf_alc, clf_fpa, clf_fpl):
        self.clf_alc = clf_alc
        self.clf_fpa = clf_fpa
        self.clf_fpl = clf_fpl
        
    def __call__(self, df, thres=0.75):
        self.df = df
        
        for col in self.cols:
            self.df[col] = 0 
        
        self.thres = thres
        
        self._make_alcohol_predictions()
        self._make_firstperson_predictions()
        self._make_firstpersonlevel_predictions()
        
        return self.df
        
    
    def _make_alcohol_predictions(self):
        predictions_alc = self.clf_alc.predict_proba(self.df)
        self.df["predict_alc"] = predictions_alc[:,1]
    
    def _make_firstperson_predictions(self):
        filter_alc = self.df.predict_alc > self.thres

        # predict only on subset of the data, makes things way faster
        predict_fpa = self.clf_fpa.predict_proba(self.df[filter_alc])
        self.df.loc[filter_alc, "predict_fpa|alc"] = predict_fpa[:,1]

        # compute a marginal using the product rule
        self.df["predict_fpa"] = self.df["predict_alc"] * self.df["predict_fpa|alc"]
        
    def _make_firstpersonlevel_predictions(self):
        filter_alc = self.df.predict_alc > self.thres
        
        # predict only on subset of the data, makes things way faster
        predict_fpl = self.clf_fpl.predict_proba(self.df[filter_alc])

        # convert it to a named dataframe
        predict_fpl = pd.DataFrame(
            predict_fpl, 
                 columns=[
                "predict_present|fpa", 
                "predict_future|fpa", 
                "predict_past|fpa"],
            index=self.df[filter_alc].index)
        
        marginal_firstperson = self.df[filter_alc]["predict_fpa"]
        
        # for each conditional level generate a marginal
        for col in predict_fpl.columns:
            col_marginal = col.split("|")[0]
            predict_fpl[col_marginal] = predict_fpl[col] * marginal_firstperson
            
        self.df = self.df.join(predict_fpl).fillna(0)

clf = PredictionTransformer(clf_alc, clf_fpa, clf_fpl)

labeld_dataframe = clf(pd.read_csv(twitter_data[3]))

labeld_dataframe[
    ["predict_fpa", "predict_alc", "predict_present", "predict_future", "predict_past", "text"]
][(labeld_dataframe.predict_fpa	 > .70) 
  | (labeld_dataframe.predict_present > .60)
  | (labeld_dataframe.predict_past > .60)
  | (labeld_dataframe.predict_future > .60)
 ].sample(10)

