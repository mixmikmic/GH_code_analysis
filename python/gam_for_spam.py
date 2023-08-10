get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

import pandas as pd

colnames = ["word_freq_make", "word_freq_address", "word_freq_al", "word_freq_3d", "word_freq_our",
            "word_freq_over", "word_freq_remove", "word_freq_internet", "word_freq_order", "word_freq_mail",
            "word_freq_receive", "word_freq_will", "word_freq_people", "word_freq_report", "word_freq_addresses",
            "word_freq_free", "word_freq_business", "word_freq_email", "word_freq_you", "word_freq_credit",
            "word_freq_your", "word_freq_font", "word_freq_000", "word_freq_money", "word_freq_hp", "word_freq_hpl",
            "word_freq_george", "word_freq_650", "word_freq_lab", "word_freq_labs", "word_freq_telnet",
            "word_freq_857", "word_freq_data", "word_freq_415", "word_freq_85", "word_freq_technology",
            "word_freq_1999", "word_freq_parts", "word_freq_pm", "word_freq_direct", "word_freq_cs",
            "word_freq_meeting", "word_freq_original", "word_freq_project", "word_freq_re", "word_freq_edu",
            "word_freq_table", "word_freq_conference", "char_freq_;", "char_freq_(", "char_freq_[", "char_freq_!",
            "char_freq_$", "char_freq_#", "capital_run_length_average", "capital_run_length_longest",
            "capital_run_length_total", "category"]

spam = pd.read_csv('spambase/spambase.data', header=None, names=colnames)

spam.shape

from pygam import LogisticGAM
from pygam.utils import generate_X_grid

allpredictors = colnames[0:-1]
keptpredictors = allpredictors[0:5] # modify this line to include more/less predictors

X = spam[keptpredictors].values 
y = spam[colnames[-1]].values

gam = LogisticGAM().gridsearch(X, y) # WARNING : this may be very long to run

XX = generate_X_grid(gam)

plt.figure(figsize = (12, 3))
#fig, axs = plt.subplots(1, 5)
titles = keptpredictors

for i, title in enumerate(titles[0:4]):
    pdep, confi = gam.partial_dependence(XX, feature=i+1, width=.95)
    ax = plt.subplot(1, 4, i+1)
    ax.plot(XX[:, i], pdep)
    #ax.plot(XX[:, i], confi, c='r', ls='--')
    ax.set_title(title)



