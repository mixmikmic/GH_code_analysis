## implemented LINK with solver='lbfgs'
from __future__ import division
from matplotlib.backends.backend_pdf import PdfPages
import os

## relevant libraries
execfile('../functions/python_libraries.py')

## data file-paths
execfile('../functions/define_paths.py')

## processing datasets
execfile('../functions/create_adjacency_matrix.py') 
execfile('../functions/create_directed_adjacency_matrix.py')
execfile('../functions/compute_homophily.py')
execfile('../functions/compute_monophily.py')


execfile('../functions/parsing.py')
execfile('../functions/mixing.py')

## code for gender prediction 
execfile('../functions/LINK.py')
execfile('../functions/majority_vote.py')


execfile('../functions/benchmark_classifier.py')

## gender preference distribution
execfile('../functions/compute_null_distribution.py')

for f in listdir(fb100_file):
    if f.endswith('.mat'):
        tag = f.replace('.mat', '')
        school = 'Amherst41'  # 'Amherst41' 'MIT8' # we report results for MIT8 and Amherst41
        if (tag==school):
            print tag
            input_file = path_join(fb100_file, f)
            A, metadata = parse_fb100_mat_file(input_file)

            adj_matrix_tmp = A.todense()
            gender_y_tmp = metadata[:,1] #gender
                
            gender_dict = create_dict(range(len(gender_y_tmp)), gender_y_tmp)
                
            (gender_y, adj_matrix_gender) = create_adj_membership(
                                    nx.from_numpy_matrix(adj_matrix_tmp), # graph
                                                           gender_dict,   # dictionary
                                                           0,             # val_to_drop, gender = 0 is missing
                                                           'yes',         # delete_na_cols, ie completely remove NA nodes from graph
                                                           0,             # diagonal
                                                           None,          # directed_type
                                                           'gender')      # gender
            
            gender_y = np.array(map(np.int,gender_y)) ## need np.int for machine precisions reasons

F_fb_label = 1
M_fb_label = 2

print len(gender_y)
print np.sum(gender_y == 1)
print np.sum(gender_y == 2)

#F
in_F_degree = adj_matrix_gender[gender_y==F_fb_label,] * np.matrix((gender_y==F_fb_label)+0).T
total_F_degree = np.sum(adj_matrix_gender[gender_y==F_fb_label,] ,1)
h_F = np.mean(in_F_degree)/np.mean(total_F_degree)

#M
in_M_degree = adj_matrix_gender[gender_y==M_fb_label,] * np.matrix((gender_y==M_fb_label)+0).T
total_M_degree = np.sum(adj_matrix_gender[gender_y==M_fb_label,] ,1)
h_M = np.mean(in_M_degree)/np.mean(total_M_degree)

print h_F - np.mean(gender_y==F_fb_label)
print ''
print h_M - np.mean(gender_y==M_fb_label)

print monophily_index_overdispersion_Williams(adj_matrix_gender, gender_y)

class_values = np.unique(gender_y)
print class_values

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

## rerun here for k=1 and then k=2
k=2 # 1 2
adj_amherst_k= np.matrix(adj_matrix_gender)**k
adj_amherst_k[range(adj_amherst_k.shape[0]),range(adj_amherst_k.shape[0])]=0 ## remove self-loops
                    
nonzero_idx1 = np.array((np.sum(adj_amherst_k[gender_y==class_values[0],:],1)!=0).T)[0]
nonzero_idx2 = np.array((np.sum(adj_amherst_k[gender_y==class_values[1],:],1)!=0).T)[0]
mv_g1 = (adj_amherst_k[gender_y==class_values[0],:] * np.matrix((gender_y==class_values[0])+0).T)[nonzero_idx1]/np.sum(adj_amherst_k[gender_y==class_values[0],:],1)[nonzero_idx1]
mv_g2 = (adj_amherst_k[gender_y==class_values[1],:] * np.matrix((gender_y==class_values[1])+0).T)[nonzero_idx2]/np.sum(adj_amherst_k[gender_y==class_values[1],:],1)[nonzero_idx2]

y_score = np.array(np.concatenate((1-mv_g1,mv_g2))).T[0]

y_test = np.concatenate((np.repeat(class_values[0],len(mv_g1)),
                     np.repeat(class_values[1],len(mv_g2))))                    

tn, fp, fn, tp =confusion_matrix(label_binarize(y_test,np.unique(y_test)), 
                 (y_score>(np.mean(gender_y==2)))+0).ravel()
print (tn, fp, fn, tp)

precision2, recall2, thresholds2 = precision_recall_curve(label_binarize(y_test, np.unique(y_test)), 
                                              y_score)
from sklearn.metrics import average_precision_score
average_precision2 = average_precision_score(label_binarize(y_test, np.unique(y_test)), 
                                            y_score)
get_ipython().magic('matplotlib inline')
plt.step(recall2, precision2, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall2, precision2, step='post', alpha=0.2,
                 color='b')


plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title(str(k)+'hop:' + str(k)+ '-class Precision-Recall curve: AP={0:0.2f}'.format(
          average_precision2))
plt.show()

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(label_binarize(y_test, np.unique(y_test)), 
                                 y_score)#, 
                                 #pos_label=1)
print len(fpr)
print len(thresholds)

if k == 1:
    get_ipython().magic('matplotlib inline')
    ax = plt.subplot(111)
    ax.axvline(np.mean(gender_y==2),
              color='gray',
              alpha=0.8)
    ax.annotate('accuracy threshold', xy=(np.mean(gender_y==2)+0.17, 0.85), 
                     color='gray', alpha=1, size=12)

    ax.scatter(1-thresholds, fpr,
               color='blue', alpha = 0.2)
    ax.annotate('FPR', xy=(0.1, 0.85), 
                     color='blue', alpha=1, size=12)
    ax.scatter(1-thresholds, tpr,
               color='red', alpha = 0.2)
    ax.annotate('TPR', xy=(0.1, 0.9), 
                     color='red', alpha=1, size=12)
    ax.set_xlabel('Classifier Threshold Cutoff')
    ax.set_ylabel('Performance Metric')
    ax.set_title(school+ ', ' + str(k)+'-MV')

    ax.set_ylim([-0.01, 1.01])
    #ax.set_xlim([-1.01, 1.51])
    ax.set_xlim([-0.51, 1.01])


    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    pp = PdfPages('../../figures/' + school +'_tpr_fpr_k'+str(k)+'.pdf')
    pp.savefig()
    pp.close()
    #plt.show()

if k==2:
    get_ipython().magic('matplotlib inline')
    ax = plt.subplot(111)
    ax.axvline(np.mean(gender_y==2),
              color='gray',
              alpha=0.8)
    ax.annotate('accuracy threshold', xy=(np.mean(gender_y==2)+0.09, 0.85), 
                     color='gray', alpha=1, size=12)

    ax.scatter(1-thresholds, fpr,
               color='blue', alpha = 0.2)
    ax.annotate('FPR', xy=(0.1, 0.85), 
                     color='blue', alpha=1, size=12)
    ax.scatter(1-thresholds, tpr,
               color='red', alpha = 0.2)
    ax.annotate('TPR', xy=(0.1, 0.9), 
                     color='red', alpha=1, size=12)
    ax.set_xlabel('Classifier Threshold Cutoff')
    ax.set_ylabel('Performance Metric')
    ax.set_title(school+ ', ' + str(k)+'-MV')

    ax.set_ylim([-0.01, 1.01])
    ax.set_xlim([-0.51, 1.01])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    pp = PdfPages('../../figures/' + school +'_tpr_fpr_k'+str(k)+'.pdf')
    pp.savefig()
    pp.close()
    #plt.show()

print sklearn.metrics.f1_score(label_binarize(y_test, np.unique(y_test)), 
                               np.array(y_score>0.5)+0, #labels=None, pos_label=1, 
                               average='binary', sample_weight=None)

## rerun above part so that precision2, recall2 correspond with k=2
k=1
adj_amherst_k= np.matrix(adj_matrix_gender)**k
adj_amherst_k[range(adj_amherst_k.shape[0]),range(adj_amherst_k.shape[0])]=0 ## remove self-loops
                    
nonzero_idx1 = np.array((np.sum(adj_amherst_k[gender_y==class_values[0],:],1)!=0).T)[0]
nonzero_idx2 = np.array((np.sum(adj_amherst_k[gender_y==class_values[1],:],1)!=0).T)[0]
mv_g1 = (adj_amherst_k[gender_y==class_values[0],:] * np.matrix((gender_y==class_values[0])+0).T)[nonzero_idx1]/np.sum(adj_amherst_k[gender_y==class_values[0],:],1)[nonzero_idx1]
mv_g2 = (adj_amherst_k[gender_y==class_values[1],:] * np.matrix((gender_y==class_values[1])+0).T)[nonzero_idx2]/np.sum(adj_amherst_k[gender_y==class_values[1],:],1)[nonzero_idx2]

y_score = np.array(np.concatenate((1-mv_g1,mv_g2))).T[0]

y_test = np.concatenate((np.repeat(class_values[0],len(mv_g1)),
                     np.repeat(class_values[1],len(mv_g2))))                    

tn, fp, fn, tp =confusion_matrix(label_binarize(y_test,np.unique(y_test)), 
                 (y_score>(np.mean(gender_y==2)))+0).ravel()
print tn, fp, fn, tp 

precision, recall, thresholds = precision_recall_curve(label_binarize(y_test, np.unique(y_test)), 
                                              y_score)
from sklearn.metrics import average_precision_score
average_precision = average_precision_score(label_binarize(y_test, np.unique(y_test)), 
                                            y_score)

plt.step(recall, precision, color='r', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='r')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('1hop: 2-class Precision-Recall curve: AP={0:0.2f}'.format(
          average_precision))

print sklearn.metrics.f1_score(label_binarize(y_test, np.unique(y_test)), 
                               np.array(y_score>0.5)+0, #labels=None, pos_label=1, 
                               average='binary', sample_weight=None)

plt.step(recall2, precision2, color='b', alpha=0.8,
         where='post')
plt.fill_between(recall2, precision2, step='post', alpha=0.8,
                 color='b')
plt.text(0.8,0.95,'2-hopMV',color = 'b')


plt.step(recall, precision, color='r', alpha=0.8,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.8,
                 color='red')

plt.text(0.8,0.9,'1-hopMV',color = 'red')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
#plt.show()
pp = PdfPages('../../figures/' + school +'_precision_recall.pdf')
pp.savefig()
pp.close()

px1 = precision[0:len(precision)-1]
rx1 = recall[0:len(recall)-1]

px2 = precision2[0:len(precision2)-1]
rx2 = recall2[0:len(recall2)-1]

idx2 = np.array(range(0,len(rx2), 100))
idx = np.array(range(0,len(rx1), 100))

np.mean(gender_y==2)

if school == 'Amherst41':
    idx_accuracy = np.array(thresholds==0.5)#np.mean(gender_y==2))
    print np.sum(idx_accuracy)
    idx_accuracy2 = np.array(thresholds2==0.5)#np.mean(gender_y==2))
    print np.sum(idx_accuracy2)
if school == 'MIT8':
    idx_accuracy = np.array(thresholds==0.6)#np.mean(gender_y==2))
    print np.sum(idx_accuracy)
    idx_accuracy2 = np.array(thresholds2==0.6)#np.mean(gender_y==2))
    print np.sum(idx_accuracy2)


get_ipython().magic('matplotlib inline')
ax = plt.subplot(111)
ax.step(recall2, precision2, color='b', alpha=0.1,
         where='post')
ax.fill_between(recall2, precision2, step='post', alpha=0.8,
                 color='b')
ax.text(0.8,1,'2-hopMV',color = 'b')
ax.text(0.8,0.9,'Accuracy',color = 'black')
ax.scatter(0.78,0.91, color = 'black',s=42, alpha = 1)



ax.step(recall, precision, color='r', alpha=0.1,
         where='post')
ax.fill_between(recall, precision, step='post', alpha=0.8,
                 color='red')

ax.scatter(rx2[idx_accuracy2], px2[idx_accuracy2], color = 'black',s=42, alpha = 1)
ax.scatter(rx1[idx_accuracy], px1[idx_accuracy], color = 'black', s=42, alpha = 1)


ax.text(0.8,0.95,'1-hopMV',color = 'red')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_ylim([0.0, 1.05])
ax.set_xlim([0.0, 1.0])
ax.set_title(school)

#plt.show()
pp = PdfPages('../../figures/' + school +'_precision_recall.pdf')
pp.savefig()
pp.close()



