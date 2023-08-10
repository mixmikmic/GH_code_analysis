get_ipython().magic('pylab inline')
from time import time

# create an n by n array
n=1000
a=ones([n,n])

get_ipython().run_cell_magic('time', '', '# Scan column by column\ns=0;\nfor i in range(n):\n    s+=sum(a[:,i])')

get_ipython().run_cell_magic('time', '', '## Scan row by row\ns=0;\nfor i in range(n):\n    s+=sum(a[i,:])')

def sample_run_times(T,k=10):
    """ compare the time to sum an array row by row vs column by column
        T: the sizes of the matrix, [10**e for e in T]
        k: the number of repetitions of each experiment
    """
    all_times=[]
    for e in T:
        n=int(10**e)
        print '\r',n,
        a=np.ones([n,n])
        times=[]

        for i in range(k):    
            t0=time()
            s=0;
            for i in range(n):
                s+=sum(a[:,i])
            t1=time()
            s=0;
            for i in range(n):
                s+=sum(a[i,:])
            t2=time()
            times.append({'row minor':t1-t0,'row major':t2-t1})
        all_times.append({'n':n,'times':times})
    return all_times

#example run
sample_run_times([1,2],k=1)

all_times=sample_run_times(np.arange(1,3.01,0.001),k=1)

n_list=[a['n'] for a in all_times]
ratios=[a['times'][0]['row minor']/a['times'][0]['row major'] for a in all_times]

figure(figsize=(15,10))
plot(n_list,ratios)
grid()
xlabel('n')
ylabel('ratio or running times')
title('time ratio as a function of size of array');

k=100
all_times=sample_run_times(np.arange(1,2.81,0.01),k=k)
_n=[]
_row_major_mean=[]
_row_major_std=[]
_row_major_std=[]
_row_minor_mean=[]
_row_minor_std=[]
_row_minor_min=[]
_row_minor_max=[]

for times in all_times:
    _n.append(times['n'])
    row_major=[a['row major'] for a in times['times']]
    row_minor=[a['row minor'] for a in times['times']]
    _row_major_mean.append(np.mean(row_major))
    _row_major_std.append(np.std(row_major))
    
    _row_minor_mean.append(np.mean(row_minor))
    _row_minor_std.append(np.std(row_minor))
    _row_minor_min.append(np.min(row_minor))
    _row_minor_max.append(np.max(row_minor))

_row_major_mean=np.array(_row_major_mean)
_row_major_std=np.array(_row_major_std)
_row_minor_mean=np.array(_row_minor_mean)
_row_minor_std=np.array(_row_minor_std)

figure(figsize=(15,10))
plot(_n,_row_major_mean,label='row major mean')
plot(_n,_row_major_mean-_row_major_std,label='row major mean-std')
plot(_n,_row_major_mean+_row_major_std,label='row major mean+std')
plot(_n,_row_minor_mean,label='row minor mean')
plot(_n,_row_minor_mean-_row_minor_std,label='row minor mean-std')
plot(_n,_row_minor_mean+_row_minor_std,label='row minor mean+std')
plot(_n,_row_minor_min,label='row minor min among %d'%k)
plot(_n,_row_minor_max,label='row minor max among %d'%k)
legend()
grid()



