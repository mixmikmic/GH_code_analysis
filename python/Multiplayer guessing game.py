# Import libraries
get_ipython().magic('matplotlib inline')
import numpy as np
import numpy.matlib
import matplotlib
import matplotlib.pyplot as plt
from IPython.html.widgets import interactive, FloatSliderWidget, fixed
from IPython.display import display

# Functions to calculate optimal trading strategies for each player
def nWin(n1,n_other,N):
    '''
    Calculate # of numbers controlled by the player who guessed 'n1'
    in a game of range 'N' in which the other guesses are in the
    array 'n_other'
    '''
    # Game is invalid if any players choose the same number
    if n1 in n_other:
        return 0
        
    nowin = []
    for no in range(len(n_other)):
        nowin = np.hstack((nowin,numCloser(n_other[no],n1,N)))
    return N - len(np.unique(nowin))


def numCloser(n1,n2,N):
    '''
    Calculate the numbers that are closer to 'n1' compared to 'n2'
    in the entire range [0,N]
    '''
    x1 = np.abs(np.arange(N) - n1)
    x2 = np.abs(np.arange(N) - n2)
    All = np.arange(N)
    return All[x1<=x2]


def closest(nps,N):
    '''
    Calculate the players (whose bets are in 'nps') that are closest
    to each number in the range [0,N]
    '''
    P = len(nps)
    closescore = np.zeros([P,N])
    for p in range(P):
        closescore[p,:] = np.abs(np.arange(N) - nps[p])
    
    closests = np.argmin(closescore,axis=0)
    for n in range(N):
        temp1 = np.min(closescore[:,n])
        if sum(closescore[:,n] == temp1) != 1:
            closests[n] = -1
    
    return closests

# Plotting functions
def plot2player(pRED,pBLUE,N):
    N = np.int(N)
    ns = np.arange(N+1)
    ns = np.hstack((-1,ns))
    closests = closest([pRED,pBLUE],N)
    closests = np.hstack((-1,closests,-1))
    closests = np.matlib.repmat(closests,2,1)
    y = np.arange(2)

    plt.figure(figsize=(24,1))
    cmap = matplotlib.colors.ListedColormap(['gray','red','blue'])
    plt.pcolor(ns, y, closests, cmap=cmap)
    for n in range(N):
        plt.plot([n,n],[0,1],'k-')

    plt.tick_params(axis='y',which='both',left='off',right='off',labelleft='off')
    plt.xticks(np.arange(N)+.5,np.arange(N))
    plt.xlim([0,N])
    plt.show()

def plot3player(pRED,pBLUE,pGREEN,N):
    N = np.int(N)
    ns = np.arange(N+1)
    ns = np.hstack((-1,ns))
    closests = closest([pRED,pBLUE,pGREEN],N)
    closests = np.hstack((-1,closests,-1))
    closests = np.matlib.repmat(closests,2,1)
    y = np.arange(2)

    plt.figure(figsize=(24,1))
    cmap = matplotlib.colors.ListedColormap(['gray','red','blue','green'])
    plt.pcolor(ns, y, closests, cmap=cmap)
    for n in range(N):
        plt.plot([n,n],[0,1],'k-')

    plt.tick_params(axis='y',which='both',left='off',right='off',labelleft='off')
    plt.xticks(np.arange(N)+.5,np.arange(N))
    plt.xlim([0,N])
    plt.show()

def plot4player(pRED,pBLUE,pGREEN,pYELLOW,N):
    N = np.int(N)
    ns = np.arange(N+1)
    ns = np.hstack((-1,ns))
    closests = closest([pRED,pBLUE,pGREEN,pYELLOW],N)
    closests = np.hstack((-1,closests,-1))
    closests = np.matlib.repmat(closests,2,1)
    y = np.arange(2)

    plt.figure(figsize=(24,1))
    cmap = matplotlib.colors.ListedColormap(['gray','red','blue','green','yellow'])
    plt.pcolor(ns, y, closests, cmap=cmap)
    for n in range(N):
        plt.plot([n,n],[0,1],'k-')

    plt.tick_params(axis='y',which='both',left='off',right='off',labelleft='off')
    plt.xticks(np.arange(N)+.5,np.arange(N))
    plt.xlim([0,N])
    plt.show()

# Experiment parameters
P = 2 #Number of players
N = 20 #Range of integers in the game

# Calculate the number of numbers controlled by each player for each possible combination of choices
W = np.zeros([N,N,2])
for n1 in range(N):
    for n2 in range(N):
        W[n1,n2,0] = nWin(n1,[n2],N)
        W[n1,n2,1] = nWin(n2,[n1],N)

# Calculate the optimal number choices for P1 and P2
p1_bestres_eachn = np.zeros(N)
for n in range(N):
    p1_bestres_eachn[n] = N - np.max(np.squeeze(W[n,:,1]))
p1bestN = np.argmax(p1_bestres_eachn)
p2bestN = np.argmax(np.squeeze(W[p1bestN,:,1]))
p2bestW = W[p1bestN,p2bestN,1]
p1bestW = W[p1bestN,p2bestN,0]

# Display results of strategically played game
print 'P1 chooses' , p1bestN , '... Controls' , np.int(p1bestW), '/', N, 'numbers'
print 'P2 chooses' , p2bestN , '... Controls' , np.int(p2bestW), '/', N, 'numbers'

# Plot the numbers controlled by each player
w=interactive(plot2player,
              pRED  = FloatSliderWidget(min = 0, max = N-1, step = 1, value = p1bestN),
              pBLUE = FloatSliderWidget(min = 0, max = N-1, step = 1, value = p2bestN),
              N = fixed(N))
display(w)

# Experiment parameters
P = 3 #Number of players
N = 50 #Range of integers in the game

# Calculate the number of numbers controlled by each player for each possible combination of choices
W = np.zeros([N,N,N,P])
for n1 in range(N):
    for n2 in range(N):
        for n3 in range(N):
            W[n1,n2,n3,0] = nWin(n1,[n2,n3],N)
            W[n1,n2,n3,1] = nWin(n2,[n1,n3],N)
            W[n1,n2,n3,2] = nWin(n3,[n1,n2],N)

# Calculate the best choices for P2 and P3 for every possible choice by P1
p2bestN_1 = np.zeros(N)
p3bestN_1 = np.zeros(N)
p1bestW = np.zeros(N)
p2bestW = np.zeros(N)
p3bestW = np.zeros(N)
for n1 in range(N):
    # Calculate the best possible P2W for every n2
    p2_bestres_eachn = np.zeros(N)
    for n2 in range(N):
        c_bestp3 = np.argmax(np.squeeze(W[n1,n2,:,2]))
        p2_bestres_eachn[n2] = W[n1,n2,c_bestp3,1]
        
    # Choose best P2N, followed by P3N
    p2bestN_1[n1] = np.argmax(p2_bestres_eachn)
    p3bestN_1[n1] = np.argmax(np.squeeze(W[n1,p2bestN_1[n1],:,2]))
    
    # Calculate winnings for all players
    p1bestW[n1] = W[n1,p2bestN_1[n1],p3bestN_1[n1],0]
    p2bestW[n1] = W[n1,p2bestN_1[n1],p3bestN_1[n1],1]
    p3bestW[n1] = W[n1,p2bestN_1[n1],p3bestN_1[n1],2]

# Calculate the best option for P1 and its instantiations
p1bestN = np.argmax(p1bestW)
p2bestN = p2bestN_1[p1bestN]
p3bestN = p3bestN_1[p1bestN]
p1bestW = W[p1bestN,p2bestN,p3bestN,0]
p2bestW = W[p1bestN,p2bestN,p3bestN,1]
p3bestW = W[p1bestN,p2bestN,p3bestN,2]


# Display optimal game
print 'P1 chooses' , p1bestN , '... Controls' , np.int(p1bestW), '/', N, 'numbers'
print 'P2 chooses' , p2bestN , '... Controls' , np.int(p2bestW), '/', N, 'numbers'
print 'P3 chooses' , p3bestN , '... Controls' , np.int(p3bestW), '/', N, 'numbers'

# Plot the numbers controlled by each player
w=interactive(plot3player,
              pRED  = FloatSliderWidget(min = 0, max = N-1, step = 1, value = p1bestN),
              pBLUE = FloatSliderWidget(min = 0, max = N-1, step = 1, value = p2bestN),
              pGREEN = FloatSliderWidget(min = 0, max = N-1, step = 1, value = p3bestN),
              N = fixed(N))
display(w)

# Compare results of the 3-player game, range 50, for different strategies by P3 (GREEN)
bets = np.round(np.array([[37,12,11],[37,12,36],[37, 9,36]])) #N=50

# Calculate controlled numbers by each player for each betting round
p1bestW2 = np.zeros(np.shape(bets)[0])
p2bestW2 = np.zeros(np.shape(bets)[0])
p3bestW2 = np.zeros(np.shape(bets)[0])
for c in range(np.shape(bets)[0]):
    p1bestW2[c] = W[bets[c][0],bets[c][1],bets[c][2],0] / N
    p2bestW2[c] = W[bets[c][0],bets[c][1],bets[c][2],1] / N
    p3bestW2[c] = W[bets[c][0],bets[c][1],bets[c][2],2] / N

plt.figure(figsize = (8,4))
plt.bar([0,4,8] ,p1bestW2,width=1,color='red')
plt.bar([1,5,9] ,p2bestW2,width=1,color='blue')
plt.bar([2,6,10],p3bestW2,width=1,color='green')
plt.xticks([1.5,5.5,9.5],['No bias','GREEN biased against RED','BLUE and GREEN collude'])
plt.xlim([-.5,11.5])
plt.ylim([0,.51])
plt.ylabel('Probability of winning game')
plt.show()

# Experiment parameters
P = 4 #Number of players
N = 20 #Range of integers in the game

# Calculate the number of numbers controlled by each player for each possible combination of choices
W = np.zeros([N,N,N,N,P])
for n1 in range(N):
    for n2 in range(N):
        for n3 in range(N):
            for n4 in range(N):
                W[n1,n2,n3,n4,0] = nWin(n1,[n2,n3,n4],N)
                W[n1,n2,n3,n4,1] = nWin(n2,[n1,n3,n4],N)
                W[n1,n2,n3,n4,2] = nWin(n3,[n1,n2,n4],N)
                W[n1,n2,n3,n4,3] = nWin(n4,[n1,n2,n3],N)

# Calculate the best choices for P3,4 for every possible choice by P1,P2
p2bestN_1 = np.zeros(N)
p3bestN_1 = np.zeros([N,N])
p4bestN_1 = np.zeros([N,N])
p1bestW = np.zeros([N,N])
p2bestW = np.zeros([N,N])
p3bestW = np.zeros([N,N])
p4bestW = np.zeros([N,N])

p1_bestres = np.zeros(N)
for n1 in range(N):
    p2_bestres = np.zeros(N)
    
    for n2 in range(N):
        p3_bestres = np.zeros(N)
        
        # Calculate winnings for different choices by P3 for all choices by P1 and P2
        for n3 in range(N):
            p4_bestres = np.argmax(np.squeeze(W[n1,n2,n3,:,3]))
            p3_bestres[n3] = W[n1,n2,n3,p4_bestres,2]
        
        # Calculate best choice by P3 for all choices by P1 and P2
        p3bestN_1[n1,n2] = np.argmax(p3_bestres)
        p4bestN_1[n1,n2] = np.argmax(np.squeeze(W[n1,n2,p3bestN_1[n1,n2],:,3]))
        
        # Calculate winnings for different choices by P2 for all choices by P1
        p2_bestres[n2] = W[n1,n2,p3bestN_1[n1,n2],p4bestN_1[n1,n2],1]
    
    # Calculate best choice by P2 for all choices by P1
    p2bestN_1[n1] = np.argmax(p2_bestres)
    p1_bestres[n1] = W[n1,p2bestN_1[n1],p3bestN_1[n1,p2bestN_1[n1]],p4bestN_1[n1,p2bestN_1[n1]],0]

# Calculate the best option for P1 and subsequent results
p1bestN = np.argmax(p1_bestres)
p2bestN = p2bestN_1[p1bestN]
p3bestN = p3bestN_1[p1bestN,p2bestN]
p4bestN = p4bestN_1[p1bestN,p2bestN]
p1bestW = W[p1bestN,p2bestN,p3bestN,p4bestN,0]
p2bestW = W[p1bestN,p2bestN,p3bestN,p4bestN,1]
p3bestW = W[p1bestN,p2bestN,p3bestN,p4bestN,2]
p4bestW = W[p1bestN,p2bestN,p3bestN,p4bestN,3]

# Display optimal game
print 'P1 chooses' , p1bestN , '... Controls' , np.int(p1bestW), '/', N, 'numbers'
print 'P2 chooses' , p2bestN , '... Controls' , np.int(p2bestW), '/', N, 'numbers'
print 'P3 chooses' , p3bestN , '... Controls' , np.int(p3bestW), '/', N, 'numbers'
print 'P4 chooses' , p4bestN , '... Controls' , np.int(p4bestW), '/', N, 'numbers'

# Plot the numbers controlled by each player
w=interactive(plot4player,
              pRED  = FloatSliderWidget(min = 0, max = N-1, step = 1, value = p1bestN),
              pBLUE = FloatSliderWidget(min = 0, max = N-1, step = 1, value = p2bestN),
              pGREEN = FloatSliderWidget(min = 0, max = N-1, step = 1, value = p3bestN),
              pYELLOW = FloatSliderWidget(min = 0, max = N-1, step = 1, value = p4bestN),
              N = fixed(N))
display(w)

