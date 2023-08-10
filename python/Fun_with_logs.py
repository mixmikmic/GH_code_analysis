ls

from las import LASReader
import matplotlib.pyplot as plt
import numpy as np
get_ipython().magic('matplotlib inline')

f = 'E-38.las'
well = LASReader(f , null_subs=np.nan)

data = well.data2d
data[8000:8003]

depth = data[:,0]
depth[:10]

well.curves.display()

well.start, well.stop

gr = np.vstack((data[:,0], data[:,14])).T
print gr
gr.shape

gr = gr[~np.isnan(gr[:,1])]
print gr

fig = plt.figure(figsize=(16,2))
plt.plot(gr[:,0], gr[:,1])
plt.show()

d = gr[:,0]
g = gr[:,1]
binno = 31
norm = binno * (g-np.amin(g))/(np.amax(g)-np.amin(g))
bins = np.arange(binno)
dig = np.digitize(norm, bins, right=True)
np.amax(dig), np.amin(dig)

fig = plt.figure(figsize=(16,2))
plt.plot(d, dig)
plt.show()

plt.figure(figsize=(9,9))

for p in range(9):
    p += 1
    glcm = np.zeros((binno,binno))

    for i,v in enumerate(dig):
        try:
            glcm[v,dig[i + 2*p]] += 1
        except:
            continue
            
    glcm /= np.sum(glcm)
    plt.subplot(3,3,p)
    plt.axis('off')
    plt.title(np.mean(glcm))
    plt.imshow(np.sqrt(glcm), interpolation='nearest')
    
plt.show()

plt.figure(figsize=(3,3))

glcm = np.zeros((binno,binno))

for p in range(9):
    p += 1
    for i,v in enumerate(dig):
        try:
            glcm[v,dig[i + 2*p]] += 1
        except:
            continue

glcm /= np.sum(glcm)
            
plt.axis('off')
plt.title('Step DOWN, Avg scale')
plt.imshow(np.sqrt(glcm), interpolation='nearest')
    
plt.show()

print "First 4 grey values in GLCM..."
print glcm[:4,:4]

plt.figure(figsize=(6,3))

glcm_down = np.zeros((binno,binno))

for p in range(9):
    p += 1
    for i,v in enumerate(dig):
        try:
            glcm_down[v,dig[i - 2*p]] += 1
        except:
            continue

glcm_down /= np.sum(glcm_down)
            
plt.subplot(1,2,1)
plt.axis('off')
plt.title('Step UP')
plt.imshow(np.sqrt(glcm_down), interpolation='nearest')
plt.subplot(1,2,2)
plt.axis('off')
plt.title('Difference')
plt.imshow(glcm_down.T - glcm, interpolation='nearest')
    
plt.show()

subdig = dig[2000:5000]

stat = np.zeros_like(subdig, dtype=float)

for i in np.arange(subdig.size - 100):
    
    i += 50
        
    glcm = np.zeros((binno,binno))

    for p in range(9):
        p += 1
        for j,v in enumerate(subdig[i-49:i+50]):
            j += i # 'correct' j to dig index
            try:
                glcm[v,subdig[j + 2*p]] += 1
            except:
                continue
                
    stat[i] = np.var(glcm)

plt.figure(figsize=(3,9))
depths = np.arange(2000,4798)
plt.plot(subdig[51:-151], depths, 'lightgray')
plt.plot(stat[51:-151], depths)
plt.gca().invert_yaxis()
plt.show()

subdig = dig[2000:5000]
scales = 24

stat = np.zeros((subdig.size,scales), dtype=float)

for i in np.arange(subdig.size - 100):
    
    i += 50
        
    for p in range(scales):
        glcm = np.zeros((binno,binno))

        for j,v in enumerate(subdig[i-49:i+50]):
            j += i # 'correct' j to dig index
            try:
                glcm[v,subdig[j + 4*p + 1]] += 1
            except:
                continue
                
        stat[i,p] = np.var(glcm)

stretch = np.repeat(stat,10,1)
plt.figure(figsize=(2,20))
plt.imshow(stretch)
plt.show()

subg = g[2000:5000]
scl = 100
maxvar = np.amax(stat,1)
minvar = np.amin(stat,1)
maxind = np.argmax(stat,1)

plt.figure(figsize=(5,25))
depths = np.arange(2000,4798)
plt.plot(subg[51:-151], depths, 'r')
plt.plot(maxind[51:-151], depths, 'lightgray')
# plt.plot(stat[51:-151,0]*scl, depths, 'r')
# plt.plot(stat[51:-151,23]*scl, depths, 'r')
#plt.plot(maxvar[51:-151]*scl, depths, 'g')
#plt.plot(minvar[51:-151]*scl, depths, 'g')
plt.fill_betweenx(depths, minvar[51:-151]*scl, maxvar[51:-151]*scl, lw=0, alpha=0.3)
plt.gca().invert_yaxis()
plt.show()



