import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import permutation_test_score
from gensim.models.phrases import Phrases
from gensim.models.phrases import Phraser
from nltk import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# A function to extract the city name from the string giving the location
def get_job_city(job_loc):
    job_loc_split = str(job_loc).split(",")
    return(job_loc_split[0])

# A function to extract the state from the string giving the location
def get_job_state(job_loc):
    job_loc_split = str(job_loc).split(",")
    if len(job_loc_split) > 1:
        return(str(job_loc_split[1]).split()[0])
    else:
        return("")
    
# A class to fit a first-order phrase model to a series of job titles
class PhraseBigram(BaseEstimator, TransformerMixin):
    def __init__(self, punct_list, stop_list):
        self.punct_list = punct_list
        self.stop_list = stop_list

    def fit(self, X, y = None):
        # Based on code I saw here: https://www.reddit.com/r/learnmachinelearning/comments/5onknw/python_nlp_need_advice_about_gensim_phrasesphraser/
        # Initialize stemmer
        from gensim.models.phrases import Phrases
        from gensim.models.phrases import Phraser
        from nltk.stem.lancaster import LancasterStemmer
        from nltk import word_tokenize
        lancaster_stemmer = LancasterStemmer()
        # Set lists of characters/words to exclude
        punct_list = self.punct_list
        stop_list = self.stop_list
        # Get sentence stream from titles
        bigram_stream = [[lancaster_stemmer.stem(i.lower()) for i in word_tokenize(sent) if i not in punct_list and i not in stop_list] for sent in list(X)]
        bigram = Phraser(Phrases(bigram_stream, min_count=3, threshold=3, delimiter=b' '))
        self.bigram = bigram
        return(self)
    
    def transform(self, X):
        from gensim.models.phrases import Phrases
        from gensim.models.phrases import Phraser
        from nltk.stem.lancaster import LancasterStemmer
        from nltk import word_tokenize
        lancaster_stemmer = LancasterStemmer()
        punct_list = self.punct_list
        stop_list = self.stop_list
        bigram = self.bigram
        x_list = []
        for j in X:
            doc = [lancaster_stemmer.stem(i) for i in word_tokenize(j) if i not in punct_list and i not in stop_list]
            x_list.append("-".join(bigram[doc]))
            
        return(pd.Series(x_list))

# Function to use as custom tokenizer for results of PhraseBigram.transform
def dash_tokenizer(sent):
    return(sent.split("-"))

#Get the job metadata and job descriptions that were scraped previously from Indeed in a dataframe
job_metadata = pd.read_csv("job_ad_metadata_v2.csv")
job_descriptions = pd.read_csv("job_ad_descriptions_v2.csv")

# Get series with the city name by applying get_job_city to each location string
job_cities = job_metadata.job_loc.apply(get_job_city)

# Get series with the state by applying get_job_state to each location string
job_states = job_metadata.job_loc.apply(get_job_state)
job_states_lim = job_states

# Give states with reasonable numbers of jobs their own class, group all other as "Other"
job_states_lim[np.logical_not((job_states == "CA") | (job_states == "MA") | (job_states == "NY") | (job_states == "WA") | (job_states == "TX") | (job_states == "NJ") | (job_states == "VA"))] = "Other"

# Intialize the steps in the model and place in a pipeline. Get CV estimates of performance
title_phrase_model = PhraseBigram([".", "-", "_", "!", "?", "[", "]", "(", ")", "%", "$", "&", ",", "/", ":", "–", " "], ["the", "of", "a", "CA"])
job_title_count_vec = CountVectorizer(max_df = 0.95, min_df = 5, stop_words = 'english', tokenizer = dash_tokenizer)
title_logit_model = LogisticRegression()
title_pipeline = Pipeline([('phrase_model', title_phrase_model), ('count_vec', job_title_count_vec), ("logit_model", title_logit_model)])
np.mean(cross_val_score(title_pipeline, job_metadata.job_title, job_states_lim, cv = StratifiedKFold(5), scoring = "accuracy"))

# Permutation test for relevance of fitted model
score, permutation_scores, pvalue = permutation_test_score(title_pipeline, job_metadata.job_title, job_states_lim.values, scoring="accuracy", cv = StratifiedKFold(5), n_permutations = 100)

# Borrow code from http://scikit-learn.org/stable/auto_examples/feature_selection/plot_permutation_test_for_classification.html#sphx-glr-auto-examples-feature-selection-plot-permutation-test-for-classification-py
# Plot a histogram of permutated score as well as the observed score
plt.hist(permutation_scores, 20, label='Permutation scores')
ylim = plt.ylim()
plt.plot(2 * [score], ylim, '--g', linewidth=3,
         label='Classification Score'
         ' (pvalue %s)' % pvalue)

plt.ylim(ylim)
plt.legend()
plt.xlabel('Accuracy')
plt.show()

# Fit pipeline on full data set
title_pipe_fit = title_pipeline.fit(job_metadata.job_title, job_states_lim)

# Store components of of the fitted pipeline
title_logit_model = title_pipe_fit.named_steps["logit_model"]
title_phrase_model = title_pipe_fit.named_steps["phrase_model"]
title_count_model = title_pipe_fit.named_steps["count_vec"]

# Transform full dataset and get feature names
title_transform_dense = title_count_model.fit_transform(title_phrase_model.fit_transform(job_metadata.job_title)).todense()
title_transform_dense_names = title_pipeline.get_params()["count_vec"].get_feature_names()
job_title_count_dense = pd.DataFrame(title_transform_dense)
job_title_count_dense.columns = title_transform_dense_names

# Get top features positively associated with the job being in CA
job_title_count_dense.columns[title_logit_model.coef_[0] > 1]

# Get top features negatively associated with the job being in CA
job_title_count_dense.columns[title_logit_model.coef_[0] < -1]

# Visualize logistic model coefficients by state
title_multi_logit_coefs = pd.DataFrame(title_logit_model.coef_)
title_multi_logit_coefs.columns = title_transform_dense_names
fig, ax = plt.subplots()
ax.scatter(title_multi_logit_coefs["machin learn"], title_multi_logit_coefs["stat"])
for i, (x, y) in enumerate(zip(title_multi_logit_coefs["machin learn"], title_multi_logit_coefs["stat"])):
    ax.annotate(list(title_logit_model.classes_)[i], (x, y), xytext=(5, 5), textcoords='offset points')
plt.xlabel("machin learn")
plt.ylabel("stat")
plt.show()

# Create state label as above for jobs for which I was able to scrape a job description
job_descriptions_omit = job_descriptions.dropna()
job_cities_descr = job_descriptions_omit.job_loc.apply(get_job_city)
job_states_descr = job_descriptions_omit.job_loc.apply(get_job_state)
job_states_lim_descr = job_states_descr
job_states_lim_descr[np.logical_not((job_states_descr == "CA") | (job_states_descr == "MA") | (job_states_descr == "NY") | (job_states_descr == "WA") | (job_states_descr == "TX") | (job_states_descr == "NJ") | (job_states_descr == "VA"))] = "Other"

# Initialize models for job descriptions
descr_phrase_model = PhraseBigram([".", "-", "_", "!", "?", "[", "]", "(", ")", "%", "$", "&", ",", "/", ":", "–", " ", ">", "#", "@"], ["the", "of", "a", "CA"])
descr_count_vec = CountVectorizer(max_df = 0.95, min_df = 20, stop_words = 'english', tokenizer = dash_tokenizer)
descr_pipeline = Pipeline([('phrase_model', descr_phrase_model), ('count_vec', descr_count_vec)])

# Fit feature models for job descriptions and get transformed features
descr_pipe_fit = descr_pipeline.fit(job_descriptions_omit.description, job_states_lim_descr)
descr_phrase_model = descr_pipe_fit.named_steps["phrase_model"]
descr_count_model = descr_pipe_fit.named_steps["count_vec"]
descr_transform_dense = descr_count_model.fit_transform(descr_phrase_model.fit_transform(job_descriptions_omit.description)).todense()
descr_transform_dense_names = descr_pipe_fit.get_params()["count_vec"].get_feature_names()
descr_count_dense = pd.DataFrame(descr_transform_dense)
descr_count_dense.columns = descr_transform_dense_names

# Refit the models for the titles that also have descriptions
title_descr_dense = title_count_model.transform(title_phrase_model.transform(job_descriptions_omit.job_title)).todense()
title_descr_dense = pd.DataFrame(title_descr_dense)
title_descr_dense.columns = title_transform_dense_names

stat_sas_tab = pd.crosstab(title_descr_dense["stat"], descr_count_dense["sas"] == 0)
ml_sas_tab = pd.crosstab(title_descr_dense["machin learn"], descr_count_dense["sas"] == 0)
ds_sas_tab = pd.crosstab(title_descr_dense["dat sci"] > 0, descr_count_dense["sas"] == 0)
stat_r_tab = pd.crosstab(title_descr_dense["stat"], descr_count_dense["r"] == 0)
ml_r_tab = pd.crosstab(title_descr_dense["machin learn"], descr_count_dense["r"] == 0)
ds_r_tab = pd.crosstab(title_descr_dense["dat sci"] > 0, descr_count_dense["r"] == 0)
stat_python_tab = pd.crosstab(title_descr_dense["stat"], descr_count_dense["python"] == 0)
ml_python_tab = pd.crosstab(title_descr_dense["machin learn"], descr_count_dense["python"] == 0)
ds_python_tab = pd.crosstab(title_descr_dense["dat sci"] > 0, descr_count_dense["python"] == 0)

stat_proportions = [stat_sas_tab.iloc[1, 0]/(stat_sas_tab.iloc[1, 0] + stat_sas_tab.iloc[1, 1]), stat_r_tab.iloc[1, 0]/(stat_r_tab.iloc[1, 0] + stat_r_tab.iloc[1, 1]), stat_python_tab.iloc[1, 0]/(stat_python_tab.iloc[1, 0] + stat_python_tab.iloc[1, 1])]
ml_proportions = [ml_sas_tab.iloc[1, 0]/(ml_sas_tab.iloc[1, 0] + ml_sas_tab.iloc[1, 1]), ml_r_tab.iloc[1, 0]/(ml_r_tab.iloc[1, 0] + ml_r_tab.iloc[1, 1]), ml_python_tab.iloc[1, 0]/(ml_python_tab.iloc[1, 0] + ml_python_tab.iloc[1, 1])]
ds_proportions = [ds_sas_tab.iloc[1, 0]/(ds_sas_tab.iloc[1, 0] + ds_sas_tab.iloc[1, 1]), ds_r_tab.iloc[1, 0]/(ds_r_tab.iloc[1, 0] + ds_r_tab.iloc[1, 1]), ds_python_tab.iloc[1, 0]/(ds_python_tab.iloc[1, 0] + ds_python_tab.iloc[1, 1])]

program_labels = ["sas", "r", "python"]
title_labels = ["stat", "machin learn", "dat sci"]

plt.subplot(131)
axes = plt.gca()
axes.set_ylim([0.0, 0.5])
plt.bar(np.arange(len(program_labels)), stat_proportions, align='center', color = "black")
plt.xticks(np.arange(len(program_labels)), program_labels)
plt.ylabel('Proportion of jobs mentioning')
plt.title('stat')

plt.subplot(132)
axes = plt.gca()
axes.set_ylim([0.0, 0.5])
plt.bar(np.arange(len(program_labels)), ml_proportions, align='center', color = "red")
plt.xticks(np.arange(len(program_labels)), program_labels)
plt.title('machin learn')

plt.subplot(133)
axes = plt.gca()
axes.set_ylim([0.0, 0.5])
plt.bar(np.arange(len(program_labels)), ds_proportions, align='center', color = "blue")
plt.xticks(np.arange(len(program_labels)), program_labels)
plt.title('dat sci')

plt.tight_layout()

