import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

mc_sem = pd.read_csv('semantria and meaningcloud.csv')
bing = pd.read_csv('Bing_Output.csv')

mc_sem.head()

bing.head()

Bing_analysis = bing.analysis_label_for_neg.tolist() + bing.analysis_label_for_pos.tolist()
True_label = [-1] * 1000 + [1]*1000

confusion_matrix(True_label,Bing_analysis)

print classification_report(True_label,Bing_analysis)

semantria_analysis = mc_sem.Semantriaanalysis_for_neg.tolist() + mc_sem.Semantriaanalysis_for_pos.tolist()
mc_analysis = mc_sem.mcanalysis_for_neg.tolist() + mc_sem.mcanalysis_for_pos.tolist()

print classification_report(True_label_2,semantria_analysis)

confusion_matrix(True_label_2,semantria_analysis)

print classification_report(True_label_2,mc_analysis)

combined_analysis = []
for i in range(len(Bing_analysis)):
    x = Bing_analysis[i]
    y = semantria_analysis[i]
    z = mc_analysis[i]
    combined_analysis.append((x+y+z)/3)

print classification_report(True_label,combined_analysis)

combined_analysis = []
for i in range(len(Bing_analysis)):
    x = Bing_analysis[i]*0.6
    y = semantria_analysis[i]*0.15
    z = mc_analysis[i]*0.15
    combined_analysis.append((x+y+z))

combined_analysisad = []
for i in combined_analysis:
    if i < 0:
        i = -1
    else:
        i = 1
    combined_analysisad.append(i)

print classification_report(True_label,combined_analysisad)

