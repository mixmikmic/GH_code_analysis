get_ipython().magic('run include/utils.py')
get_ipython().magic('matplotlib inline')

def plot_hist(data,xlines=None,title="",xlabel="",ylabel=""):
    (K,T) = data.shape
    fig = plt.figure(figsize=(20,6))
    ax = fig.gca()
    y,x = np.mgrid[slice(0, K+1, 1),slice(0,T+1, 1)]
    cm = ax.pcolormesh(x, y, data)
    ax.hold(True)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(cm)
    fig.canvas.draw()
    
def plot_matrix(X, title='Title', xlabel='xlabel', ylabel='ylabel', figsize=None):
    if figsize is None:
        plt.figure(figsize=(25,6))
    else:
        plt.figure(figsize=figsize)
    plt.imshow(X, interpolation='none', vmax=np.max(X), vmin=0, aspect='auto')
    plt.colorbar()
    plt.set_cmap('gray_r')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

import scipy.stats as st

# X ~ TV
def nmf_fit_icm(X, rank, fixed_T=None, at = 10, av = 0.2, MAX_ITER=1000):

    [I, J] = X.shape

    # T
    if fixed_T is None:
        bt = 1.0
        At = at * np.ones((I, rank))
        Bt = bt * np.ones((I, rank))
        Ct = At/Bt
        T = np.random.gamma(at, bt / at, (I, rank))
    else:
        T = np.copy(fixed_T)

    # V
    bv = 1.0
    Av = av * np.ones((rank, J))
    Bv = bv * np.ones((rank, J))
    Cv = Av/Bv
    V = np.random.gamma(av, bv / av, (rank, J))

    M = np.ones(X.shape)
    KL = np.zeros(MAX_ITER)

    T_trans = T.transpose()
    for it in range(MAX_ITER):
        TV = np.dot(T,V)
        if fixed_T is None:
            V_trans = V.transpose()
            T = (At + T * (np.dot((X / TV), V_trans))) / (Ct + np.dot(M, V_trans))
        V = (Av + V * (np.dot(T_trans, (X / TV)))) / (Cv + np.dot(T_trans, M))

        KL[it] = st.entropy(X.flat[:], np.dot(T, V).flat[:])

    # Normalize W:
    s = T.sum(axis=0)
    T = T/s
    
    return [T, V, KL]


# X ~ TV
def nmf_fit_multiplicative(X, rank, MAX_ITER=1000):
    [I, J] = X.shape
    
    # initialization
    T = np.random.dirichlet(np.ones(I),rank).transpose()
    V = np.random.dirichlet(np.ones(rank),J).transpose()
    KL = np.zeros(MAX_ITER)
    
    for it in range(MAX_ITER):
        # update H
        Zw = np.tile(T.sum(axis = 0, keepdims = True).transpose(), J)
        numerator = np.dot(T.transpose(), np.divide(X, np.dot(T, V)))
        V = np.multiply(V, np.divide(numerator, Zw))

        # update W
        Zh = np.tile(V.sum(axis = 1, keepdims = True), I).transpose()
        numerator = np.dot(np.divide(X, np.dot(T, V)), V.transpose())
        T = np.multiply(T, np.divide(numerator, Zh))

        # calculate D(V||WH)
        KL[it] = st.entropy(X.flat[:], np.dot(T, V).flat[:])

    # Normalize W:
    s = T.sum(axis=0)
    T = T/s
    V = V * s.reshape((-1,1))
    
    
    return T, V, KL

I = 6
J = 1000
R = 4

T_orig = normalize(np.eye(I)+1e-1,axis=0)
V_orig = np.empty((I,0))
for i in range(R):
    tmp = np.random.dirichlet(np.ones(I)).reshape((I,1))*10
    V_sub = np.tile(tmp,(1,int(J/R))) + np.random.random((I,int(J/R)))
    V_orig = np.hstack((V_orig, V_sub))

X = np.dot(T_orig,V_orig)


plot_matrix(T_orig, title='T_orig')
plot_matrix(V_orig, title='V_orig')
plot_matrix(X, title='Data')


Ts, Vs, KLs = nmf_fit_icm(X, rank=R, MAX_ITER=100)
plot_matrix(Ts, title='Learn Bases')

