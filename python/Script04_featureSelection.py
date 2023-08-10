import pandas as pd
import numpy as np
from pandas import DataFrame

from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import (GridSearchCV, LeaveOneGroupOut)
from sklearn.metrics import (precision_score, recall_score, f1_score, accuracy_score)
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import MinMaxScaler

df_merge_tobii_tf = pd.read_csv("features_labels_tobii.csv")
df_merge_webgazer_sac = pd.read_csv("features_labels_webgazer.csv")
df_merge_tobii_sac = pd.read_csv("features_labels_tobii_sd.csv")

# Remove 3 participants since they only have 1 mind-wandering report in their test
df_merge_tobii_tf = df_merge_tobii_tf[df_merge_tobii_tf.id != "Anon06"]
df_merge_tobii_tf = df_merge_tobii_tf[df_merge_tobii_tf.id != "Anon07"]
df_merge_tobii_tf = df_merge_tobii_tf[df_merge_tobii_tf.id != "Anon13"]

df_merge_webgazer_sac = df_merge_webgazer_sac[df_merge_webgazer_sac.id != "Anon06"]
df_merge_webgazer_sac = df_merge_webgazer_sac[df_merge_webgazer_sac.id != "Anon07"]
df_merge_webgazer_sac = df_merge_webgazer_sac[df_merge_webgazer_sac.id != "Anon13"] 

df_merge_tobii_sac = df_merge_tobii_sac[df_merge_tobii_sac.id != "Anon06"]
df_merge_tobii_sac = df_merge_tobii_sac[df_merge_tobii_sac.id != "Anon07"]
df_merge_tobii_sac = df_merge_tobii_sac[df_merge_tobii_sac.id != "Anon13"] 

df_merge_list = [df_merge_tobii_tf,
                 df_merge_webgazer_sac,
                 df_merge_tobii_sac]

list_id_report = list(df_merge_tobii_tf['id'].unique())

# Define different features running in the experiments
feature_index_local = [10,11,12,13,14,15,16,17,
                       27,28,29,30,31,32,33,34,35,36,37,38,39,40,41]
feature_index_local_dict = {"feature_name": "Local Features", 
                            "feature_index": feature_index_local}
feature_index_global = [18,19,20,21,22,23,24,25,26,
                        42,43,44,45,46,47,48,49,50,
                        51,52,53,54,55,56,57,58,59,
                        60,61,62,63,64,65,66,67]
feature_index_global_dict = {"feature_name": "Global Features", 
                             "feature_index": feature_index_global}
feature_index_all = range(10, 68)
feature_index_all_dict = {"feature_name": "All Features", 
                          "feature_index": feature_index_all}
featuren_list = [feature_index_all_dict, 
                 feature_index_global_dict, 
                 feature_index_local_dict]

# Define pipelines
sm = SMOTE(random_state=48)

pipe_tobii_tf_all = {"pipe_name": "Tobii Data + Tobii Filter + All Feature",
                     "feature_dict": feature_index_all_dict
                    }
pipe_tobii_tf_global = {"pipe_name": "Tobii Data + Tobii Filter + Global Feature",
                        "feature_dict": feature_index_global_dict
                       }
pipe_tobii_tf_local = {"pipe_name": "Tobii Data + Tobii Filter + Local Feature",
                       "feature_dict": feature_index_local_dict
                      }


pipe_webgazer_sac_all = {"pipe_name": "WebGazer Data + Saccade Detection + All Feature",
                         "feature_dict": feature_index_all_dict
                        }
pipe_webgazer_sac_global = {"pipe_name": "WebGazer Data + Saccade Detection + Global Feature",
                            "feature_dict": feature_index_global_dict
                           }
pipe_webgazer_sac_local = {"pipe_name": "WebGazer Data + Saccade Detection + Local Feature",
                           "feature_dict": feature_index_local_dict
                          }


pipe_tobii_sac_all = {"pipe_name": "Tobii Data + Saccade Detection + All Feature",
                      "feature_dict": feature_index_all_dict
                     }
pipe_tobii_sac_global = {"pipe_name": "Tobii Data + Saccade Detection + Global Feature",
                         "feature_dict": feature_index_global_dict
                        }
pipe_tobii_sac_local = {"pipe_name": "Tobii Data + Saccade Detection + Local Feature",
                        "feature_dict": feature_index_local_dict
                       }

pipe_list_list = [[pipe_tobii_tf_all, 
                   pipe_tobii_tf_global, 
                   pipe_tobii_tf_local], 
                  [pipe_webgazer_sac_all, 
                   pipe_webgazer_sac_global, 
                   pipe_webgazer_sac_local],
                  [pipe_tobii_sac_all, 
                   pipe_tobii_sac_global, 
                   pipe_tobii_sac_local]]

def rank_to_dict(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x, 2), ranks)
    return dict(zip(names, ranks ))

# Data Type
for i in range(0, len(df_merge_list)):
    df_merge = df_merge_list[i]
    pipe_list = pipe_list_list[i]
    
    for j in range(0, len(pipe_list)):
        
        feature_index = pipe_list[j]["feature_dict"]["feature_index"]
        feature_name = df_merge.columns[feature_index]
    
        print "-----------------------------------"
        print pipe_list[j]["pipe_name"]
        print "-----------------------------------"

        data = df_merge
        
        # data preparation
        X = data.ix[:, feature_index].fillna(value=0)
        y = list(data.ix[:, 4])

        f, pval  = f_classif(X, y)
        rank = rank_to_dict(f, feature_name)
        rank_index_sorted = sorted(rank, key=rank.get)
        rank_index_sorted.reverse()
        for i in rank_index_sorted:
            print i + ": " + str(rank[i])

import pandas as pd
import numpy as np
from pandas import DataFrame

from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import (GridSearchCV, LeaveOneGroupOut)
from sklearn.metrics import (precision_score, recall_score, f1_score, accuracy_score)
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE

df_merge_tobii_tf = pd.read_csv("features_labels_tobii.csv")
df_merge_webgazer_sac = pd.read_csv("features_labels_webgazer.csv")
df_merge_tobii_sac = pd.read_csv("features_labels_tobii_sd.csv")

# Remove 3 participants since they only have 1 mind-wandering report in their test
df_merge_tobii_tf = df_merge_tobii_tf[df_merge_tobii_tf.id != "Anon06"]
df_merge_tobii_tf = df_merge_tobii_tf[df_merge_tobii_tf.id != "Anon07"]
df_merge_tobii_tf = df_merge_tobii_tf[df_merge_tobii_tf.id != "Anon13"]

df_merge_webgazer_sac = df_merge_webgazer_sac[df_merge_webgazer_sac.id != "Anon06"]
df_merge_webgazer_sac = df_merge_webgazer_sac[df_merge_webgazer_sac.id != "Anon07"]
df_merge_webgazer_sac = df_merge_webgazer_sac[df_merge_webgazer_sac.id != "Anon13"] 

df_merge_tobii_sac = df_merge_tobii_sac[df_merge_tobii_sac.id != "Anon06"]
df_merge_tobii_sac = df_merge_tobii_sac[df_merge_tobii_sac.id != "Anon07"]
df_merge_tobii_sac = df_merge_tobii_sac[df_merge_tobii_sac.id != "Anon13"] 

df_merge_list = [df_merge_tobii_tf,
                 df_merge_webgazer_sac,
                 df_merge_tobii_sac]

list_id_report = list(df_merge_tobii_tf['id'].unique())

# Define different features running in the experiments
feature_index_local = [10,11,12,13,14,15,16,17,
                       27,28,29,30,31,32,33,34,35,36,37,38,39,40,41]
feature_index_local_dict = {"feature_name": "Local Features", 
                            "feature_index": feature_index_local}
feature_index_global = [18,19,20,21,22,23,24,25,26,
                        42,43,44,45,46,47,48,49,50,
                        51,52,53,54,55,56,57,58,59,
                        60,61,62,63,64,65,66,67]
feature_index_global_dict = {"feature_name": "Global Features", 
                             "feature_index": feature_index_global}
feature_index_all = range(10, 68)
feature_index_all_dict = {"feature_name": "All Features", 
                          "feature_index": feature_index_all}
featuren_list = [feature_index_all_dict, 
                 feature_index_global_dict, 
                 feature_index_local_dict]

# Define pipelines
pipe_tobii_tf_all = {"pipe_name": "Tobii Data + Tobii Filter + All Feature",
                     "clf": LogisticRegression(C=10, 
                                               tol=0.1,
                                               class_weight='balanced',
                                               penalty='l1'),
                     "feature_dict": feature_index_all_dict
                    }
pipe_tobii_tf_global = {"pipe_name": "Tobii Data + Tobii Filter + Global Feature",
                        "clf": LogisticRegression(C=0.1,
                                                  tol=0.1, 
                                                  class_weight='balanced', 
                                                  penalty='l1'),
                        "feature_dict": feature_index_global_dict
                       }
pipe_tobii_tf_local = {"pipe_name": "Tobii Data + Tobii Filter + Local Feature",
                       "clf": LogisticRegression(C=100, 
                                                 tol=0.1, 
                                                 class_weight='balanced', 
                                                 penalty='l2'),
                       "feature_dict": feature_index_local_dict
                      }


pipe_webgazer_sac_all = {"pipe_name": "WebGazer Data + Saccade Detection + All Feature",
                         "clf": LogisticRegression(C=0.1, 
                                                   tol=0.01, 
                                                   class_weight='balanced', 
                                                   penalty='l2'), 
                         "feature_dict": feature_index_all_dict
                        }
pipe_webgazer_sac_global = {"pipe_name": "WebGazer Data + Saccade Detection + Global Feature",
                            "clf": LogisticRegression(C=1, 
                                                      tol=0.1, 
                                                      class_weight='balanced', 
                                                      penalty='l1'),
                            "feature_dict": feature_index_global_dict
                           }
pipe_webgazer_sac_local = {"pipe_name": "WebGazer Data + Saccade Detection + Local Feature",
                           "clf": LogisticRegression(C=100, 
                                                     tol=0.1, 
                                                     class_weight='balanced', 
                                                     penalty='l1'),
                           "feature_dict": feature_index_local_dict
                          }


pipe_tobii_sac_all = {"pipe_name": "Tobii Data + Saccade Detection + All Feature",
                      "clf": LogisticRegression(C=100, 
                                                tol=0.1, 
                                                class_weight='balanced', 
                                                penalty='l2'), 
                      "feature_dict": feature_index_all_dict
                     }
pipe_tobii_sac_global = {"pipe_name": "Tobii Data + Saccade Detection + Global Feature",
                         "clf": LogisticRegression(C=1, 
                                                   tol=0.1,
                                                   class_weight='balanced', 
                                                   penalty='l1'),
                         "feature_dict": feature_index_global_dict
                        }
pipe_tobii_sac_local = {"pipe_name": "Tobii Data + Saccade Detection + Local Feature",
                        "clf": LogisticRegression(C=1, 
                                                  tol=0.0001,
                                                  class_weight='balanced', 
                                                  penalty='l2'),
                        "feature_dict": feature_index_local_dict
                       }

pipe_list_list = [[pipe_tobii_tf_all, 
                   pipe_tobii_tf_global, 
                   pipe_tobii_tf_local], 
                  [pipe_webgazer_sac_all, 
                   pipe_webgazer_sac_global, 
                   pipe_webgazer_sac_local],
                  [pipe_tobii_sac_all, 
                   pipe_tobii_sac_global, 
                   pipe_tobii_sac_local]
                 ]

def rank_to_dict(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x, 2), ranks)
    return dict(zip(names, ranks ))

# Data Type
for i in range(0, len(df_merge_list)):
    df_merge = df_merge_list[i]
    pipe_list = pipe_list_list[i]
    
    for j in range(0, len(pipe_list)):
        
        feature_index = pipe_list[j]["feature_dict"]["feature_index"]
        feature_name = df_merge.columns[feature_index]
    
        print "-----------------------------------"
        print pipe_list[j]["pipe_name"]
        print "-----------------------------------"

        data = df_merge
        
        # data preparation
        X = data.ix[:, feature_index].fillna(value=0)
        y = list(data.ix[:, 4])

        # we define the pipeline 
        clf = pipe_list[j]["clf"]

        # inner cv, model selection by gridsearch
        rfe = RFE(estimator=clf, n_features_to_select=10, step=1)
        rfe.fit(X, y)

        rank = rank_to_dict(map(float, rfe.ranking_), feature_name, order=-1)
        rank_index_sorted = sorted(rank, key=rank.get)
        rank_index_sorted.reverse()
        for i in rank_index_sorted:
            print i + ": " + str(rank[i])



