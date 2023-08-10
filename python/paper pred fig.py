import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import nengo
from nengo import spa
get_ipython().magic('matplotlib inline')

column_labels = ["3+3", "3+5", "5+3", "4+4", "1+4", "2+4", "4+1", "4+2"]

#np.load("data/paperslow_count_data.npz")
af = np.load("data/multpred2_learning_data.npz")
vo_load = np.load("data/multpred2_learning_vocab.npz")
print(af.keys())
print(vo_load.keys())

vo = spa.Vocabulary(10)
for key, val in zip(vo_load['keys'], vo_load['vecs']):
    vo.add(key, val)

recall = af['p_recall']
print(recall.shape)

for r_i in range(run_num):
    plt.plot(spa.similarity(recall[r_i][360:], vo))
plt.legend(vo.keys, bbox_to_anchor=(1.3,1))

tmp = spa.similarity(recall[0][s_win:], vo)
plt.plot(tmp[::30])

for r_i in range(run_num):
    simi = spa.similarity(recall[r_i][s_win:], vo)
    resimi = simi.reshape((trial_num, step, dims))
    plt.plot(resimi[0])

run_num = 25
trial_num = 8
dims = 9
step = 30
max_res = np.zeros((run_num, trial_num))

for r_i in range(run_num):
    simi = spa.similarity(recall[r_i][s_win:], vo).reshape((trial_num, step, dims))
    max_dim = np.argmax(np.sum(simi, axis=1), axis=1)
    
    sing_dim = np.zeros((trial_num, step))
    for t_i in range(trial_num):
        sing_dim[t_i] = simi[t_i, :, max_dim[t_i]]
    
    plt.plot(sing_dim.flatten())
    max_res[r_i] = np.max(sing_dim, axis=1)

print(max_res.shape)
learn_res = pd.DataFrame(max_res, columns=column_labels)

sns.set_style("ticks")
ax = sns.boxplot(data=learn_res, fliersize=0)
ax.set_xticklabels(column_labels, rotation=0)

# set the face colour to white
for art in ax.artists:
    art.set_facecolor('white')

ax.set_title("Confidence of Response After Learning")
ax.set_ylabel("Confidence")
ax.set_xlabel("Addition Question Asked")
sns.despine()
fig = ax.get_figure()
fig.savefig("pred_plot.pdf", format="pdf")
print(max_res.shape)

sns.set_style("ticks")
ax = sns.barplot(data=learn_res)
ax.set_xticklabels(column_labels, rotation=0)

# set the face colour to white
for art in ax.artists:
    art.set_facecolor('white')

ax.set_title("Confidence of Response After Learning")
ax.set_ylabel("Confidence")
ax.set_xlabel("Addition Question Asked")
sns.despine()
fig = ax.get_figure()
fig.savefig("pred_plot.pdf", format="pdf")
print(max_res.shape)

sns.set_style("ticks")
ax = sns.violinplot(data=learn_res)
ax.set_xticklabels(column_labels, rotation=45)
sns.despine()
print(max_res.shape)

learn_res

new_add = pd.concat((learn_res["3+3"], learn_res["4+4"])).reset_index()
one_fam = pd.concat((learn_res["3+5"], learn_res["5+3"], learn_res["1+4"], learn_res["2+4"], learn_res["4+1"], learn_res["4+2"])).reset_index()

na1 = pd.Series(new_add[0], name="Confidence")
na2 = pd.Series(np.zeros(len(new_add), dtype=np.int8), name="Familiarity")
aa = pd.concat((na1, na2), axis=1)

of1 = pd.Series(one_fam[0], name="Confidence")
of2 = pd.Series(np.ones(len(one_fam[0]), dtype=np.int8), name="Familiarity")
bb = pd.concat((of1, of2), axis=1)

pair_res = pd.concat((aa, bb))

sns.set_style("ticks")
ax = sns.barplot(x="Familiarity", y="Confidence", data=pair_res, color="white")

ax.set_title("Confidence of Response After Learning")
ax.set_ylabel("Confidence")
ax.set_xlabel("Number of Familiar Addends")
ax.set_ylim(0.3, 0.45)
sns.despine()
fig = ax.get_figure()
fig.savefig("conf_bar_plot.pdf", format="pdf")
print(max_res.shape)

mr = pd.read_hdf("data/multpred2_09_38_51.h5")

no_fam = (mr['key'] == "3+3") | (mr['key'] == "4+4")
mr.loc[no_fam, "fam"] = 0
one_fam = (mr["key"] == "3+5") | (mr["key"] == "5+3") | (mr["key"] == "1+4") | (mr["key"] == "2+4") | (mr["key"] == "4+1") | (mr["key"] == "4+2")
mr.loc[one_fam, "fam"] = 1
mr['fam'] = mr['fam'].astype(np.int8)

print(mr.columns)

sns.set_style("ticks")
ax = sns.barplot(x="fam", y="confidence", data=mr, color="white")

ax.set_title("Accuracy of Response After Learning")
ax.set_ylabel("Accuracy")
ax.set_xlabel("Number of Familiar Addends")
ax.set_ylim(0.3, 0.5)
sns.despine()
fig = ax.get_figure()
fig.savefig("conf_bar_plot.pdf", format="pdf")

sns.set_style("ticks")
ax = sns.barplot(x="fam", y="error", data=mr, color="white")

ax.set_title("Confidence of Response After Learning")
ax.set_ylabel("Confidence")
ax.set_xlabel("Number of Familiar Addends")
#ax.set_ylim(0.3, 1)
sns.despine()

