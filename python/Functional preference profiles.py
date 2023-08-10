get_ipython().run_line_magic('matplotlib', 'inline')
from neurosynth.base.dataset import Dataset
dataset = Dataset.load("data/neurosynth_60_0.6.pkl")

from sklearn.naive_bayes import GaussianNB
from classification import RegionalClassifier
from sklearn.metrics import roc_auc_score

# Instantiate a RegionalClassifier for your image and classify
clf = RegionalClassifier(dataset, 'masks/newDMN_4regions.nii.gz', GaussianNB())
clf.classify(scoring=roc_auc_score)

clf.class_score
import pandas as pd
df = pd.DataFrame(clf.class_score)
df.to_csv('AUC.csv')

nicknames = pd.read_csv('data/v4-topics-60.txt', delimiter='\t')
nicknames['topic_name'] = nicknames.apply(lambda row: '_'.join([str(row.topic_number)] + row.top_words.split(' ')[0:3]), axis=1)
nicknames = nicknames.sort_values('topic_name')

from plotting import plot_clf_polar

cognitive_topics = ['memory','categorization','switching','inhibition','priming',
                    'social','fear','emotion','learning','reward',  
                    'decision-making','imagery','spatial','attention','WM',
                    'awareness','language','math','semantics','reading']

disease_topics = ['smoking','eating-disorder','depression','schizopherenia','adhd','autism','Alzheimer-Parkinson','ptsd']

selected_topics = ['memory','categorization','switching','inhibition','priming',
                    'social','fear','emotion','learning','reward',  
                    'decision-making','imagery','spatial','attention','WM',
                    'awareness','language','math','semantics','reading','action','pain',
                  'smoking','Alzheimer-Parkinson','eating-disorder','depression','autism','ptsd','adhd','schizopherenia']

formated_importances = clf.get_formatted_importances(feature_names=nicknames.nickname)
a, b, fig =plot_clf_polar(formated_importances,labels = cognitive_topics, max_val=1.2,label_size=21)

fig.savefig('DMN.svg')

formated_importances = formated_importances.sort_values('importance')
import pandas as pd
df = pd.DataFrame(formated_importances)
df.to_csv("importances_DMN.csv")

formated_importances = clf.get_formatted_importances(feature_names=nicknames.nickname)
_ = plot_clf_polar(formated_importances, labels = disease_topics, max_val=0.8,label_size=22)

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '1')
get_ipython().run_line_magic('aimport', 'classification')

# These are the names of each region given the manuscript
names_70 = ['MPFC','PCC','lTPJ','rTPJ'] #DMN

lor_ci = classification.bootstrap_log_odds(clf, 1000, feature_names=nicknames.nickname, 
                                           region_names = names_70, n_jobs=7)
subset_plot = lor_ci[(lor_ci.topic_name.isin(cognitive_topics))]

df = pd.DataFrame(subset_plot)
df.to_csv('profiles/DMN_cog_ci.csv')

subset_plot = lor_ci[(lor_ci.topic_name.isin(disease_topics))]
df = pd.DataFrame(subset_plot)
df.to_csv('profiles/DMN_dis_ci.csv')

from classification import permute_log_odds
lor_z = classification.permute_log_odds(clf, 1000, feature_names=nicknames.nickname, region_names = names_70)

cog_ps = lor_z[lor_z.nickname.isin(cognitive_topics)]
from statsmodels.sandbox.stats.multicomp import multipletests

reject, p_corr, a, a1 = multipletests(cog_ps.p, alpha=0.01, method='fdr_tsbky')

cog_ps['p_corr_01'] = p_corr # Adjusted p-value
cog_ps['reject_01'] = (cog_ps.p_corr_01<0.05) & (cog_ps.lor_z > 0) # Was the null hypothesis rejected?

import pandas as pd
df = pd.DataFrame(cog_ps)
df.to_csv("profiles/DMN_cog.csv")


dis_ps = lor_z[lor_z.nickname.isin(disease_topics)]
from statsmodels.sandbox.stats.multicomp import multipletests

reject, p_corr, a, a1 = multipletests(dis_ps.p, alpha=0.01, method='fdr_tsbky')

dis_ps['p_corr_01'] = p_corr # Adjusted p-value
dis_ps['reject_01'] = (dis_ps.p_corr_01<0.05) & (dis_ps.lor_z > 0)
import pandas as pd
df = pd.DataFrame(dis_ps)
df.to_csv("profiles/DMN_dis.csv")

clf = RegionalClassifier(dataset, 'images/MPFC/cluster_labels_k2.nii.gz', GaussianNB())
clf.classify(scoring=roc_auc_score)

get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from nilearn import plotting as niplt
from matplotlib.colors import ListedColormap
import numpy as np
colors = ["amber","dark blue grey"]
colors = sns.xkcd_palette(colors)
formated_importances = clf.get_formatted_importances(feature_names=nicknames.nickname)
_ = plot_clf_polar(formated_importances,labels = cognitive_topics,
                   palette = colors, max_val=1.2,label_size=21)

formated_importances = formated_importances.sort_values('importance')
import pandas as pd
df = pd.DataFrame(formated_importances)
df.to_csv("importances_MPFC.csv")

formated_importances = clf.get_formatted_importances(feature_names=nicknames.nickname)
_ = plot_clf_polar(formated_importances, labels = disease_topics, palette = colors,
                   label_size=35,max_val=0.6)

# These are the names of each region given the manuscript
names_70 = ['ventral','dorsal'] #MPFC ['posterior','ventral','dorsal']

lor_ci = classification.bootstrap_log_odds(clf, 1000, feature_names=nicknames.nickname, 
                                           region_names = names_70, n_jobs=7)
subset_plot = lor_ci[(lor_ci.topic_name.isin(cognitive_topics))]
import pandas as pd
df = pd.DataFrame(subset_plot)
df.to_csv('profiles/MPFC_cog_ci.csv')

subset_plot = lor_ci[(lor_ci.topic_name.isin(disease_topics))]
df = pd.DataFrame(subset_plot)
df.to_csv('profiles/MPFC_dis_ci.csv')

from classification import permute_log_odds
lor_z = classification.permute_log_odds(clf, 1000, feature_names=nicknames.nickname, region_names = names_70)

cog_ps = lor_z[lor_z.nickname.isin(cognitive_topics)]
from statsmodels.sandbox.stats.multicomp import multipletests

reject, p_corr, a, a1 = multipletests(cog_ps.p, alpha=0.01, method='fdr_tsbky')

cog_ps['p_corr_01'] = p_corr # Adjusted p-value
cog_ps['reject_01'] = (cog_ps.p_corr_01<0.05) & (cog_ps.lor_z > 0) # Was the null hypothesis rejected?

import pandas as pd
df = pd.DataFrame(cog_ps)
df.to_csv("profiles/MPFC_cog.csv")


dis_ps = lor_z[lor_z.nickname.isin(disease_topics)]
from statsmodels.sandbox.stats.multicomp import multipletests

reject, p_corr, a, a1 = multipletests(dis_ps.p, alpha=0.01, method='fdr_tsbky')

dis_ps['p_corr_01'] = p_corr # Adjusted p-value
dis_ps['reject_01'] = (dis_ps.p_corr_01<0.05) & (dis_ps.lor_z > 0)
import pandas as pd
df = pd.DataFrame(dis_ps)
df.to_csv("profiles/MPFC_dis.csv")

clf = RegionalClassifier(dataset, 'images/PCC/cluster_labels_k3.nii.gz', GaussianNB())
clf.classify(scoring=roc_auc_score)

colors = ["jungle green","yellow","red"]
colors = sns.xkcd_palette(colors)
formated_importances = clf.get_formatted_importances(feature_names=nicknames.nickname)
_ = plot_clf_polar(formated_importances,labels = cognitive_topics, palette = colors, 
                   max_val=1.2,label_size=21)

formated_importances = clf.get_formatted_importances(feature_names=nicknames.nickname)
_ = plot_clf_polar(formated_importances, labels = disease_topics, 
                   palette = colors,label_size=35,max_val=0.6
                  )

# These are the names of each region given the manuscript
names_70 = ['dorsal','medial','ventral'] #PCC

lor_ci = classification.bootstrap_log_odds(clf, 1000, feature_names=nicknames.nickname, 
                                           region_names = names_70, n_jobs=7)
subset_plot = lor_ci[(lor_ci.topic_name.isin(cognitive_topics))]
import pandas as pd
df = pd.DataFrame(subset_plot)
df.to_csv('profiles/PCC_cog_ci.csv')

subset_plot = lor_ci[(lor_ci.topic_name.isin(disease_topics))]
df = pd.DataFrame(subset_plot)
df.to_csv('profiles/PCC_dis_ci.csv')

from classification import permute_log_odds
lor_z = classification.permute_log_odds(clf, 1000, feature_names=nicknames.nickname, region_names = names_70)

cog_ps = lor_z[lor_z.nickname.isin(cognitive_topics)]
from statsmodels.sandbox.stats.multicomp import multipletests

reject, p_corr, a, a1 = multipletests(cog_ps.p, alpha=0.01, method='fdr_tsbky')

cog_ps['p_corr_01'] = p_corr # Adjusted p-value
cog_ps['reject_01'] = (cog_ps.p_corr_01<0.05) & (cog_ps.lor_z > 0) # Was the null hypothesis rejected?

import pandas as pd
df = pd.DataFrame(cog_ps)
df.to_csv("profiles/PCC_cog.csv")


dis_ps = lor_z[lor_z.nickname.isin(disease_topics)]
from statsmodels.sandbox.stats.multicomp import multipletests

reject, p_corr, a, a1 = multipletests(dis_ps.p, alpha=0.01, method='fdr_tsbky')

dis_ps['p_corr_01'] = p_corr # Adjusted p-value
dis_ps['reject_01'] = (dis_ps.p_corr_01<0.05) & (dis_ps.lor_z > 0)
import pandas as pd
df = pd.DataFrame(dis_ps)
df.to_csv("profiles/PCC_dis.csv")

clf = RegionalClassifier(dataset, 'images/lTPJ/cluster_labels_k3.nii.gz', GaussianNB())
clf.classify(scoring=roc_auc_score)

colors = ["clay","dodger blue","pink"]
colors = sns.xkcd_palette(colors)
formated_importances = clf.get_formatted_importances(feature_names=nicknames.nickname)
_ = plot_clf_polar(formated_importances,labels = cognitive_topics, 
                   palette = colors, label_size=21,max_val=1.2
                  )

formated_importances = clf.get_formatted_importances(feature_names=nicknames.nickname)
_ = plot_clf_polar(formated_importances, labels = disease_topics, palette = colors,label_size=35,max_val=0.6)

# These are the names of each region given the manuscript
names_70 = ['ventral','posterior','dorsal']#['ventral','dorsal','medial'] #lTPJ

lor_ci = classification.bootstrap_log_odds(clf, 1000, feature_names=nicknames.nickname, 
                                           region_names = names_70, n_jobs=7)
subset_plot = lor_ci[(lor_ci.topic_name.isin(cognitive_topics))]
import pandas as pd
df = pd.DataFrame(subset_plot)
df.to_csv('profiles/lTPJ_cog_ci.csv')

subset_plot = lor_ci[(lor_ci.topic_name.isin(disease_topics))]
df = pd.DataFrame(subset_plot)
df.to_csv('profiles/lTPJ_dis_ci.csv')

from classification import permute_log_odds
lor_z = classification.permute_log_odds(clf, 1000, feature_names=nicknames.nickname, region_names = names_70)

cog_ps = lor_z[lor_z.nickname.isin(cognitive_topics)]
from statsmodels.sandbox.stats.multicomp import multipletests

reject, p_corr, a, a1 = multipletests(cog_ps.p, alpha=0.01, method='fdr_tsbky')

cog_ps['p_corr_01'] = p_corr # Adjusted p-value
cog_ps['reject_01'] = (cog_ps.p_corr_01<0.05) & (cog_ps.lor_z > 0) # Was the null hypothesis rejected?

import pandas as pd
df = pd.DataFrame(cog_ps)
df.to_csv("profiles/lTPJ_cog.csv")


dis_ps = lor_z[lor_z.nickname.isin(disease_topics)]
from statsmodels.sandbox.stats.multicomp import multipletests

reject, p_corr, a, a1 = multipletests(dis_ps.p, alpha=0.01, method='fdr_tsbky')

dis_ps['p_corr_01'] = p_corr # Adjusted p-value
dis_ps['reject_01'] = (dis_ps.p_corr_01<0.05) & (dis_ps.lor_z > 0)
import pandas as pd
df = pd.DataFrame(dis_ps)
df.to_csv("profiles/lTPJ_dis.csv")

clf = RegionalClassifier(dataset, 'images/rTPJ/cluster_labels_k2.nii.gz', GaussianNB())
clf.classify(scoring=roc_auc_score)

colors = ["lighter green","bluey purple"]
colors = sns.xkcd_palette(colors)
formated_importances = clf.get_formatted_importances(feature_names=nicknames.nickname)
_ = plot_clf_polar(formated_importances,labels = cognitive_topics, palette = colors, max_val=1.2,label_size=21)

formated_importances = clf.get_formatted_importances(feature_names=nicknames.nickname)
_ = plot_clf_polar(formated_importances, labels = disease_topics, palette = colors,label_size=22,max_val=0.6)

# These are the names of each region given the manuscript
names_70 = ['dorsal','ventral']#['ventral','dorsal','lateral'] #rTPJ

from classification import permute_log_odds
lor_z = classification.permute_log_odds(clf, 1000, feature_names=nicknames.nickname, region_names = names_70)

lor_ci = classification.bootstrap_log_odds(clf, 1000, feature_names=nicknames.nickname, 
                                           region_names = names_70, n_jobs=7)
subset_plot = lor_ci[(lor_ci.topic_name.isin(cognitive_topics))]
import pandas as pd
df = pd.DataFrame(subset_plot)
df.to_csv('profiles/rTPJ_cog_ci.csv')

subset_plot = lor_ci[(lor_ci.topic_name.isin(disease_topics))]
df = pd.DataFrame(subset_plot)
df.to_csv('profiles/rTPJ_dis_ci.csv')

cog_ps = lor_z[lor_z.nickname.isin(cognitive_topics)]
from statsmodels.sandbox.stats.multicomp import multipletests

reject, p_corr, a, a1 = multipletests(cog_ps.p, alpha=0.01, method='fdr_tsbky')

cog_ps['p_corr_01'] = p_corr # Adjusted p-value
cog_ps['reject_01'] = (cog_ps.p_corr_01<0.05) & (cog_ps.lor_z > 0) # Was the null hypothesis rejected?

import pandas as pd
df = pd.DataFrame(cog_ps)
df.to_csv("profiles/rTPJ_cog.csv")


dis_ps = lor_z[lor_z.nickname.isin(disease_topics)]
from statsmodels.sandbox.stats.multicomp import multipletests

reject, p_corr, a, a1 = multipletests(dis_ps.p, alpha=0.01, method='fdr_tsbky')

dis_ps['p_corr_01'] = p_corr # Adjusted p-value
dis_ps['reject_01'] = (dis_ps.p_corr_01<0.05) & (dis_ps.lor_z > 0)
import pandas as pd
df = pd.DataFrame(dis_ps)
df.to_csv("profiles/rTPJ_dis.csv")

















lor_ci = classification.bootstrap_log_odds(clf, 1000, feature_names=nicknames.nickname, 
                                           region_names = names_70, n_jobs=7)
subset_plot = lor_ci[(lor_ci.topic_name.isin(selected_topics))]

from matplotlib.colors import rgb2hex
from colors import l_70_colors
get_ipython().run_line_magic('load_ext', 'rpy2.ipython')

subset_plot = lor_ci[(lor_ci.topic_name.isin(cognitive_topics))]
colors_hex = [rgb2hex(c) for c in colors]
colors_hex, _ = zip(*sorted(zip(colors_hex, names_70), key=lambda tup: tup[1]))
colors_hex = list(colors_hex)

names = list(reversed(names_70))

s1 = ['reward','fear','emotion','memory','social','awareness','decision-making']
s2 = selected_topics[1:]

get_ipython().run_line_magic('Rpush', 'subset_plot')
get_ipython().run_line_magic('Rpush', 'colors_hex')
get_ipython().run_line_magic('Rpush', 'names')
get_ipython().run_line_magic('Rpush', 's2')
get_ipython().run_line_magic('Rpush', 's1')

rgb2hex(c)

get_ipython().run_cell_magic('R', '-w 450 -h 600 ', 'library(ggplot2)\nggplot(subset(subset_plot, topic_name %in% s1), aes(mean, region, color=factor(region))) + geom_point(size=2.5) + \ngeom_errorbarh(aes(xmin=low_ci, xmax=hi_ci), height=.2, size=.85) + geom_vline(xintercept = 0, alpha=.5) + facet_grid(topic_name~.) +\ntheme_bw(base_size = 17) +  theme(legend.position="none") + labs(x = \'Strength of association (LOR)\', y="") +\nscale_color_manual(values = colors_hex) + scale_y_discrete(limits=names)')



