import os
import joblib
from utils import pretty_trim, simple_split, score_top_preds, get_cmap
from collections import Counter
from scipy.sparse import vstack
import numpy as np
import chardet
from sklearn.metrics import accuracy_score
get_ipython().magic('matplotlib notebook')

get_ipython().run_cell_magic('time', '', "filename = 'models_persistence/pickle_models'\n(pretty_trim, counter, tfidf, rfe, clfs) = joblib.load(filename)")

get_ipython().run_cell_magic('time', '', "filename = 'models_persistence/final_dataset'\n(X_train_final, y_train, X_test_final, y_test) = joblib.load(filename)")

def visualize_sparse_vector(X, y, n_classes, title, n_lsa=100):
    from sklearn.manifold import TSNE
    from sklearn.decomposition import TruncatedSVD as LSA
    from time import time
    import matplotlib.pyplot as plt
    lsa = LSA(n_components=n_lsa, random_state=42)
    tsne = TSNE(n_components=2, random_state=42)
    t0 = time()
    X_lsa = lsa.fit_transform(X)
    print time() - t0, 'secs'
    print 'LSA shape:', X_lsa.shape
    t0 = time()
    X_tsne = tsne.fit_transform(X_lsa)
    print time() - t0, 'secs'
    print 't-SNE shape:', X_tsne.shape
    cmap = get_cmap(n_classes)
    plt.figure()
    plt.title(title)
    marker_list = u'.ovp3>1<_48s^*h+xD|2'
    scatters = []
    for i in xrange(n_classes):
        X_same_label = X_tsne[y == i, :]
        X_avg = X_same_label.mean(axis=0)
        scat = plt.scatter(X_same_label[:, 0], X_same_label[:, 1], s=50, c=cmap(i), marker=marker_list[i])
        plt.scatter(X_avg[0], X_avg[1], s=350, c=cmap(i), marker=marker_list[i], alpha=0.7)
        scatters.append(scat)
    plt.legend(scatters, np.arange(n_classes), loc='best')

visualize_sparse_vector(X_train_final, y_train, clfs[0].n_classes_, 'Train Set Visualization')

visualize_sparse_vector(X_test_final, y_test, clfs[0].n_classes_, 'Test Set Visualization')

str_labels = [u'0 บริหารธุรกิจ',
    u'1 ประมง',
    u'2 มนุษยศาสตร์',
    u'3 วนศาสตร์',
    u'4 วิทยาการจัดการ',
    u'5 วิทยาศาสตร์',
    u'6 วิทยาศาสตร์การกีฬา',
    u'7 วิศวกรรมศาสตร์',
    u'8 ศิลปศาสตร์และวิทยาศาสตร์',
    u'9 ศึกษาศาสตร์',
    u'10 ศึกษาศาสตร์และพัฒนศาสตร์',
    u'11 สถาปัตยกรรมศาสตร์',
    u'12 สังคมศาสตร์',
    u'13 สัตวแพทยศาสตร์',
    u'14 สิ่งแวดล้อม',
    u'15 อุตสาหกรรมเกษตร',
    u'16 เกษตร',
    u'17 เศรษฐศาสตร์',
    u'18 โครงการจัดตั้งวิทยาเขตสุพรรณบุรี',
    u'19 โครงการสหวิทยาการระดับบัณฑิตศึกษา']

# # refitting on the whole dataset
# X_old, y_old = vstack([X_train_final, X_test_final]), np.concatenate([y_train, y_test])
# clf.fit(X_old, y_old)

clf = clfs[0]
print 'score:', clf.score(X_test_final, y_test)
proba = clf.predict_proba(X_test_final)
confidence = proba.max(1)
print 'confidence:', confidence.mean(), confidence.std(), confidence.min(), confidence.max()

get_ipython().run_cell_magic('time', '', "doc_path = u'./corpus/segmented-journal' # must be a segmented doc path\ndataset_contents = []\nfilename2index = dict()\nfor i, filename in enumerate(os.listdir(doc_path)):\n    path = os.path.join(doc_path, filename)\n    filename2index[filename] = i\n    with open(path) as f:\n        content = f.read()\n#         if chardet.detect(content)['encoding'] == 'ascii':\n#             continue\n        content = content.decode('utf8')\n        dataset_contents.append(content)\nprint 'total files:', len(dataset_contents)")

get_ipython().run_cell_magic('time', '', 'for i in xrange(len(dataset_contents)):\n    dataset_contents[i] = pretty_trim(dataset_contents[i])')

get_ipython().magic('time X_new_count = counter.transform(dataset_contents)')
get_ipython().magic('time X_new_tfidf = tfidf.transform(X_new_count)')
print X_new_tfidf.shape

get_ipython().magic('time X_new_rfe = rfe.transform(X_new_tfidf)')
print X_new_rfe.shape

y_pred = clf.predict(X_new_rfe)
Counter(y_pred)

proba = clf.predict_proba(X_new_rfe)
confidence = proba.max(1)
print 'confidence:', confidence.mean(), confidence.std(), confidence.min(), confidence.max()

# filenames = ['A0906251543307343.txt', 'A1410221625383281.txt', 'A1006241011134591.txt', 'A1004071348071718.txt']
# # these are files that contain word 'department'
# indices = [filename2index[filename] for filename in filenames]
# labels = [str_labels[pred] for pred in y_pred[indices]]
# for label, prob in zip(labels, confidence[indices]):
#     print label, prob

# # what is the label of all docs that contain word 'rice' ?
# preds = []
# for i, content in enumerate(dataset_contents):
#     if u'rice' in content.split():
#         preds.append((str_labels[y_pred[i]], confidence[i]))
# preds.sort(key=lambda item: item[1], reverse=True)
# for pred in preds:
#     print pred[0], pred[1]

# [word -> class_label] mapping dictioanry
approx_label = {
#     "liber": 8,
#     "art": 8,
    "agricultur": 16,
    "agro": 16,
    "educ": 9,
#     "social": 12,
#     "fisheri": 1,
#     "manag": 4,
    "scienc": 5,
    "technolog": 5,
#     "medicin": 13,
#     "pharmaci": 13,
#     "forestri": 3,
#     "forest": 3,
    "engin": 7,
    "econom": 17,
    "architectur": 11,
#     "human": 2,
    "biotechnolog": 5,
#     "environment": 14,
#     "environ": 14,
#     "veterinari": 13,
#     "busi": 0,
#     u"ธุรกิจ": 0,
}
approx_label_detailed = {
    "liber": 8,
    "art": 8,
    "agricultur": 16,
    "agro": 16,
    "educ": 9,
    "social": 12,
    "fisheri": 1,
    "manag": 4,
    "scienc": 5,
    "technolog": 5,
    "medicin": 13,
    "pharmaci": 13,
    "forestri": 3,
    "forest": 3,
    "engin": 7,
    "econom": 17,
    "architectur": 11,
    "human": 2,
    "biotechnolog": 5,
    "environment": 14,
    "environ": 14,
    "veterinari": 13,
    "busi": 0,
    u"ธุรกิจ": 0,
}

def find_heuristic_y(approx_label):
    heuristic_y = np.zeros(len(dataset_contents), dtype=np.int32) - 1 # starts with -1 filled
    for ci in range(len(dataset_contents)):
        words = dataset_contents[ci].split()
        contexts = []
        wis = []
        for wi, word in enumerate(words):
            if u'faculti' in word or u'คณะ' in word:
                context = words[wi-3:wi+5]
                contexts.append(context)
                wis.append(wi)
                for w in context:
                    if w in approx_label:
                        heuristic_y[ci] = approx_label[w]
                        break
            if heuristic_y[ci] != -1:
                break
        if contexts: # logging
            label = str_labels[heuristic_y[ci]] if heuristic_y[ci] != -1 else 'UNKNOWN'
            print 'Document No.', ci, '(', label, ')'

            for i in range(len(contexts)):
                print 'Word No.', wis[i], ' => ',
                for w in contexts[i]:
                    if w in approx_label:
                        w = '[%s]' % w
                    print w,
                print
    return heuristic_y, heuristic_y != -1 # test data that do not have approximated label would be invalid

heuristic_y, valid_mask = find_heuristic_y(approx_label)
print 'Total Label Approximations:', np.count_nonzero(valid_mask)

heuristic_y_detailed, valid_mask_detailed = find_heuristic_y(approx_label_detailed)
print 'Total Label Approximations Detailed:', np.count_nonzero(valid_mask_detailed)

print 'Accuracy:', accuracy_score(heuristic_y[valid_mask], y_pred[valid_mask])
print Counter(heuristic_y[valid_mask])
print Counter(y_pred[valid_mask])
print
print 'Accuracy Detailed:', accuracy_score(heuristic_y_detailed[valid_mask_detailed], y_pred[valid_mask_detailed])
print Counter(heuristic_y_detailed[valid_mask_detailed])
print Counter(y_pred[valid_mask_detailed])

print "Confidence score on valid_mask:", clf.predict_proba(X_new_rfe[valid_mask]).max(1).mean()
print "Confidence score on valid_mask_detailed:", clf.predict_proba(X_new_rfe[valid_mask_detailed]).max(1).mean()

for i in range(1, 9):
    score = score_top_preds(clf, X_new_rfe[valid_mask], heuristic_y[valid_mask], i)
    print 'Accuracy score (k=%d):' % i, score
for i in range(1, 9):
    score = score_top_preds(clf, X_new_rfe[valid_mask_detailed], heuristic_y_detailed[valid_mask_detailed], i)
    print 'Accuracy score detailed (k=%d):' % i, score

for confidence in [0.5, 0.7, 0.8, 0.9, 0.95]:
    score = score_top_preds(clf, X_new_rfe[valid_mask], heuristic_y[valid_mask], confidence)
    print 'Accuracy score (k=%.2f):' % confidence, score
for confidence in [0.5, 0.7, 0.8, 0.9, 0.95]:
    score = score_top_preds(clf, X_new_rfe[valid_mask_detailed], heuristic_y_detailed[valid_mask_detailed], confidence)
    print 'Accuracy score detailed (k=%.2f):' % confidence, score

