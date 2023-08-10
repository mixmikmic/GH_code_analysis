get_ipython().magic('pylab inline')

import time
def measureRandomAccess(size,filename='',k=100000):
    """ Measure the distribution of random accesses in computer memory.
    size=size of memory block.
    filename= a file that is used as an external buffer. If filename=='' then everything is done in memory.
    k = number of times that the experiment is repeated.
    output:
    mean = the mean of T
    std = the std of T
    T = a list the contains the times of all k experiments
    """
    # Prepare buffer.
    if filename == '':
        inmem=True
        A=bytearray(size)
    else:
        inmem=False
        file=open(filename,'r+')
        
    # Read and write k times from/to buffer.
    sum=0; sum2=0
    T=np.zeros(k)
    for i in range(k):
        if (i%10000==0): print i,',',
        t=time.time()
        loc=int(rand()*size)
        if inmem:
            x=A[loc:loc+4]
            A[loc]=(i % 256)
        else:
            file.seek(loc)
            poke=file.read(1)
            file.write("test")
        d=time.time()-t
        T[i]=d
        sum += d
        sum2 += d*d
    mean=sum/k; var=(sum2/k)-mean**2; std=sqrt(var)
    return (mean,std,T)

from matplotlib.backends.backend_pdf import PdfPages
from scipy.special import erf,erfinv

def PlotTime(Tsorted,Mean,Std,Color='b',LS='-',Legend=''):
    """ plot distribution of times on a log-log scale"""
    P=arange(1,0,-1.0/k)    # probability 
    loglog(Tsorted,P,color=Color,label=Legend,linestyle=LS)                 # plot log-log of 1-CDF 
    
    grid()
    loglog([Mean,Mean],[1,0.0001],color=Color,linestyle=LS)           # vert line at mean
    Y=0.1**((m_i+1.)/2.)
    loglog([Mean,min(Mean+Std,1)],[Y,Y],color=Color,linestyle=LS) # horiz line from mean to mean + std
        
    x=arange(Mean,Mean+Std*erfinv(1.0-1.0/len(Tsorted)),Std/100)  # normal distribution 
    loglog(x,1-erf((x-Mean)/Std),color=Color,linestyle=LS)

n=1000000 # size of single block (1MB)
m_list=[1,10,100,1000] #,10000] # size of file in blocks
m=None
k=100000;
L=len(m_list)
#print 'n=%d, m=%d,k=%d, m_list=' % (n,m,k),m_list

from os.path import isfile,isdir
from os import mkdir
import os
root=os.environ['HOME']
log_root=root+'/logs'
if not isdir(log_root): mkdir(log_root)
TimeStamp=str(int(time.time()))
log_dir=log_root+'/'+TimeStamp
mkdir(log_dir)
get_ipython().magic('cd $log_dir')
stat=open('stats.txt','w')

def tee(line):
    print line
    stat.write(line+'\n')

get_ipython().system('ls -lh')

def create_file(n,m,filename='DataBlock'):
    t1=time.time()
    A=bytearray(n)
    t2=time.time()
    file=open(filename,'w')
    for i in range(m):
        file.write(A)
        if i % 100 == 0:
            print i,",",
    file.close()
    t3=time.time()
    tee('\ncreating %d byte block: %f sec, writing %d blocks %f sec' % (n,t2-t1,m,t3-t2))
    return (t2-t1,t3-t2)

mean=zeros([2,L])   #0: using disk, 1: using memory
std=zeros([2,L])
T=zeros([2,L,k])

for m_i in range(len(m_list)):
    
    m=m_list[m_i]
    (t_mem,t_disk) = create_file(n,m,filename='BlockData'+str(m))

    (mean[0,m_i],std[0,m_i],T[0,m_i]) = measureRandomAccess(n*m,filename='BlockData'+str(m),k=k)
    tee('\nFile pokes mean='+str(mean[0,m_i])+', file std='+str(std[0,m_i]))

    (mean[1,m_i],std[1,m_i],T[1,m_i]) = measureRandomAccess(n*m,k=k)
    tee('\nMemory pokes mean='+str(mean[1,m_i])+', file std='+str(std[1,m_i]))

pp = PdfPages('MemoryFigure.pdf')
figure(figsize=(6,4))

Colors='bgrcmyk'  # The colors for the plot
LineStyles=['-',':']
Legends=['F','M']

fig = matplotlib.pyplot.gcf()
fig.set_size_inches(18.5,10.5)

for m_i in range(len(m_list)):
    Color=Colors[m_i % len(Colors)]
    for Type in [0,1]:
        PlotTime(sort(T[Type,m_i]),mean[Type,m_i],std[Type,m_i],
                 Color=Color,LS=LineStyles[Type],Legend=('%dMB-' % m_list[m_i])+Legends[Type])

grid()
legend(fontsize='medium')
xlabel('delay (sec)')
ylabel('1-CDF')
pp.savefig()
pp.close()

import time

Line='Consecutive Memory writes'
print Line; stat.write(Line+'\n')
n=1000
r=np.array(range(n))
for m in [1,3,5,7,10,100,1000,10000,100000,1000000]:
    t1=time.time()
    A=np.repeat(r,m)
    t2=time.time()
    tee("array of length %d repeated %d times. total size=%6.3f MB, Time per element= %g" % (n,m,float(n*m)/1000000,(t2-t1)/float(n*m)))
A=[];r=[]
stat.close()



