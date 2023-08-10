# Some modules used in this notebook 
import numpy as np
import os
import re
import colorsys
import matplotlib.pyplot as plt

modified_kallisto_path='/data/SS_RNA_seq/Code/kalliPso'
mouse_reference='/data/SS_RNA_seq/Zeisel/reference_transcriptome/Mus_musculus.GRCm38.rel79.cdna.all.fa'

seed=100

os.system('python time_test.py -k '+modified_kallisto_path+
              ' -r '+ mouse_reference+' -h ./hisat_chr_path_list.txt -s '+str(seed))

with open('./TCC/time.time') as f: TCC=float(f.readline())
with open('./bowtie1/time.time') as f: bowtie1=float(f.readline())
with open('./hisat/time.time') as f: hisat=float(f.readline())
with open('./wc/time.time') as f: wc=float(f.readline())

get_ipython().magic('matplotlib inline')

menMeans = [float(i)/60 for i in (bowtie1,hisat,TCC,wc)]

file_list=['hisat_genome_time1.txt','kallipso_time.txt']
times=np.zeros((len(file_list),1))
cur_time=0
for ind in range(len(file_list)):
    with open ('../Zeisel_pipeline/'+file_list[ind]) as f:
        for line in f:
            line1=line.split()
            if len(line1)==2 and (line1[0]=='user' or line1[0]=='sys'):
                line2 = re.split("[ms]+", line1[1])
                cur_time += int(line2[0])*60 + float(line2[1])
    times[ind]=(float(cur_time)/3600)
    cur_time=0
    
menMeans = [300.5*float(menMeans[0])/60] + [i[0] for i in times.tolist()] + [300.5*float(menMeans[-1])/60]

# Plot
N=4
HSV_tuples = [(x*1.0/N, 0.6, 0.7) for x in range(N)]
RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
ind = np.arange(N)
width = 0.45
fig = plt.figure(figsize=(5,3))
ax = fig.add_subplot(111)
rects = ax.bar(ind, menMeans, width=0.8, color=RGB_tuples, 
               zorder=4, align='center',label='Read mapping')
methods = ['    Bowtie \n', '   HISAT \n','   kallisto\n   pseudoalign \n',
           '     Word Count',]

rects = ax.patches
ii = 0
hts=map(lambda x: ("%.2f" % x) +' h',menMeans)
for rect, ht in zip(rects, hts):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height, 
            ht, ha='center', va='bottom',fontsize=11)
    ii += 1

plt.grid()
xticks_pos = [0.65*patch.get_width() + patch.get_xy()[0] -0.2 for patch in rects]
plt.xticks(xticks_pos, methods, ha='center', rotation=0, size=10)
plt.ylim(0,np.max(menMeans)*1.2)
plt.ylabel('Runtime (core-hours)',size=12)  

plt.show()

file_list=['../Zeisel_pipeline/hisat_full_time.txt',
           '../Zeisel_pipeline/kallipso_time.txt']
n = len(file_list)
times=np.zeros((n,1))

cur_time=0
for ind in range(n):
    with open (file_list[ind]) as f:
        for line in f:
            line1=line.split()
            if len(line1)==2 and (line1[0]=='user' or line1[0]=='sys'):
                line2 = re.split("[ms]+", line1[1])
                cur_time += int(line2[0])*60 + float(line2[1])
    times[ind]=float(cur_time)
    cur_time=0

menMeans = np.round(times/3600)

ind = np.arange(n)

fig = plt.figure(figsize=(3,4))
plt.gcf().subplots_adjust(bottom=0.3)
ax = fig.add_subplot(111)
rects1 = ax.bar(ind, menMeans, width=0.6, 
                color=(RGB_tuples[1],RGB_tuples[2]),zorder=4, align='center')
methods = ['HISAT \n align \n(GRCm38)','kallisto \n pseudoalign \n(GRCm38)']
plt.grid()

plt.xticks(range(n),methods, ha='center', rotation=0, size=10)

rects = ax.patches
hts=map(lambda x: str(int(x))+' h',menMeans)
for rect, ht in zip(rects, hts):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height, 
            ht, ha='center', va='bottom',fontsize=11)
plt.xlim(-0.5,n-0.5)
plt.ylim(0,1.2*np.max(menMeans) )
plt.ylabel('Runtime (core-hour)',size=12)
plt.show()

# Time clusterings

# Run times for alignment and pseudoalignment
menMeans = [float(i)/3600 for i in (bowtie1,hisat,TCC)]

# Run times for clustering
os.system('bash time_pairwise_distances.sh')
file_list=['time_TCC.txt','time_kallisto.txt','time_UMI.txt',
           'Zeisel_new_l1_TCC.time','Zeisel_new_l1_Kallisto.time','Zeisel_new_l1_UMI.time',
           'Zeisel_new_l2_TCC.time','Zeisel_new_l2_Kallisto.time','Zeisel_new_l2_UMI.time']
times=np.zeros((len(file_list),1))
cur_time=0
for ind in range(len(file_list)):
    with open ('../Zeisel_pipeline/'+file_list[ind]) as f:
        for line in f:
            line1=line.split()
            if len(line1)==2 and (line1[0]=='user' or line1[0]=='sys'):
                line2 = re.split("[ms]+", line1[1])
                cur_time += int(line2[0])*60 + float(line2[1])
    times[ind]=(float(cur_time)/3600)
    cur_time=0

# Plot
N=len(file_list)
HSV_tuples = [(x*1.0/N, 0.6, 0.7) for x in range(N)]
RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
ind = np.arange(N)
width = 0.45
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
rects = ax.bar(ind, times, width=0.8, color=RGB_tuples, 
               zorder=4, align='center',label='Read mapping')
methods = ['TCC \n JS','trans. \n JS','genes \n JS',
           'TCC \n L1','trans. \n L1','genes \n L1',
           'TCC \n L2','trans. \n L2','genes \n L2',]

rects = ax.patches
ii = 0
hts=map(lambda x: ("%.2f" % x) +' h',times)
for rect, ht in zip(rects, hts):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height, 
            ht, ha='center', va='bottom',fontsize=11)
    ii += 1

plt.grid()
xticks_pos = [0.65*patch.get_width() + patch.get_xy()[0] -0.2 for patch in rects]
plt.xticks(xticks_pos, methods, ha='center', rotation=0, size=10)
plt.ylabel('Runtime (core-hours)',size=12)    
plt.ylim(0,1.08*np.max(times))  
plt.show()

method_names = ['UMI genes','kallisto transcripts','TCC']
flnames = ['/data/SS_RNA_seq/Code/pickled/cleaned_UMI_genes_matrix_normalised_subsample100.dat',
           '/data/SS_RNA_seq/Code/pickled/sparse_kallisto_CPM_no_bias_unnormalised_subsample100.dat',
           '../Zeisel_pipeline/Zeisel_TCC_distribution_subsample100_full.dat']
ii = 0
for f in flnames:
    i = pickle.load(file(f,'rb'))
    print method_names[ii] + ' vectors contain:'
    print ' '+ str(np.shape(i)[1]) + ' entries,'
    zz = np.zeros(3005)
    for j in range(3005):
        if scipy.sparse.issparse(i):
            zz[j] = (i.getrow(j) > 0).sum()
        else:
            zz[j] = np.sum(i[j,:] > 0)
        zz[j] = float(zz[j])
    print ' '+str(np.sum(zz)/3005)+' non-zero entries on average,'
    s = 0
    i = scipy.sparse.csc_matrix(i)
    for j in range(np.shape(i)[1]):
        if i.getcol(j).sum() == 0: s += 1
    print ' '+str(s)+' columns that sum to 0'
    ii += 1

file_list=['../Trapnell_pipeline/hisat_transcriptome_pseudo_time.txt',
           '../Trapnell_pipeline/kallisto_time.txt']
n = len(file_list)
times=np.zeros((n,1))

cur_time=0
for ind in range(n):
    with open (file_list[ind]) as f:
        for line in f:
            line1=line.split()
            if len(line1)==2 and (line1[0]=='user' or line1[0]=='sys'):
                line2 = re.split("[ms]+", line1[1])
                cur_time += int(line2[0])*60 + float(line2[1])
    times[ind]=float(cur_time)
    cur_time=0

menMeans = np.round(times/3600)

ind = np.arange(n)

fig = plt.figure(figsize=(3,4))
plt.gcf().subplots_adjust(bottom=0.3)
ax = fig.add_subplot(111)
rects1 = ax.bar(ind, menMeans, width=0.6, 
                color=(RGB_tuples[1],RGB_tuples[2]),zorder=4, align='center')
methods = ['HISAT \n align \n(GRCh38)','kallisto \n pseudoalign \n(GRCh38)']
plt.grid()

plt.xticks(range(n),methods, ha='center', rotation=0, size=10)

rects = ax.patches
hts=map(lambda x: str(int(x))+' h',menMeans)
for rect, ht in zip(rects, hts):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height, 
            ht, ha='center', va='bottom',fontsize=11)
plt.xlim(-0.5,n-0.5)
plt.ylim(0,1.2*np.max(menMeans) )
plt.ylabel('Runtime (core-hour)',size=12)
plt.show()

