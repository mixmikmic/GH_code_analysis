get_ipython().magic('run include/utils.py')
get_ipython().magic('matplotlib inline')

# see https://github.com/kirbs-/hide_code for export issues

def predict(A, lp):
    lstar = np.max(lp)
    return lstar + np.log(np.dot(A,np.exp(lp-lstar)))

def postdict(A, lp):
    lstar = np.max(lp)
    return lstar + np.log(np.dot(np.exp(lp-lstar), A))

def update(y, logB, lp):
    return logB[y,:] + lp

class HMM(object):
    def __init__(self, pi, A, B):
        # p(x_0)
        self.pi = pi
        # p(x_k|x_{k-1})
        self.A = A
        # p(y_k|x_{k})
        self.B = B
        # Number of possible latent states at each time
        self.S = pi.shape[0]
        # Number of possible observations at each time
        self.R = B.shape[0]
        self.logB = np.log(self.B + 1e-100)
        self.logA = np.log(self.A + 1e-100)
        self.logpi = np.log(self.pi)
        
    def copy(self):
        pi_ = self.pi.copy()
        A_ = self.A.copy()
        B_ = self.B.copy()
        hmm = HMM(pi_,A_,B_)
        return hmm
    
    def dump(self,filename='hmm.dump', format = '%.6f'):
        with open(filename, 'w') as f:
            f.write('%d\n%d\n' % (self.S,self.R))
            for num in self.pi:
                f.write((format+'\n') % num)  
            temp = self.A.reshape(np.product(self.A.shape), order='F')
            for num in temp:
                f.write((format+'\n') % num)  
            temp = self.B.reshape(np.product(self.B.shape), order='F')
            for num in temp:
                f.write((format+'\n') % num)  
    
    @classmethod
    def from_random_parameters(cls, S=3, R=5):
        A = np.random.dirichlet(0.7*np.ones(S),S).T
        B = np.random.dirichlet(0.7*np.ones(R),S).T
        pi = np.random.dirichlet(0.7*np.ones(S)).T
        return cls(pi, A, B)
    
    @classmethod
    def from_random_parameters_fixed_columns(cls, alphabet, S=5, R=20, V=2):
        A = np.random.dirichlet(0.7*np.ones(S),S).T
        B = np.random.dirichlet(0.7*np.ones(R),S).T
        pi = np.random.dirichlet(0.7*np.ones(S)).T
        vowels = ['a','e','ı','i','o','ö','u','ü']
        ch2int = {c:i for i,c in enumerate(alphabet)}
        ind = []
        for v in vowels: 
            if v in ch2int.keys():
                ind.append(ch2int[v])
        mask = np.ones(R,dtype=bool)
        mask[ind] = 0 # masks vowels
        for c in range(V):
            B[mask,c] = 0
        for c in range(V,S):
            B[ind,c] = 0
        B = normalize(B,axis=0)
        return cls(pi, A, B)
    
    def eval_lhood(self,y):
        log_gamma = self.forward_backward_smoother(y)
        return log_sum_exp(log_gamma[:,0])
        
    def __str__(self):
        s = "Prior:\n" + str(self.pi) + "\nA:\n" + str(self.A) + "\nB:\n" + str(self.B)
        return s
    
    def __repr__(self):
        s = self.__str__()
        return s

    def predict(self, lp):
        lstar = np.max(lp)
        return lstar + np.log(np.dot(self.A,np.exp(lp-lstar)))

    def postdict(self, lp):
        lstar = np.max(lp)
        return lstar + np.log(np.dot(np.exp(lp-lstar), self.A))

    def update(self, y, lp):
        return self.logB[y,:] + lp

    def generate_sequence(self, T=10):
        # T: Number of steps
        x = np.zeros(T,dtype='int')
        y = np.zeros(T,dtype='int')

        for t in range(T):
            if t==0:
                x[t] = randgen(self.pi)
            else:
                x[t] = randgen(self.A[:,x[t-1]])    
            y[t] = randgen(self.B[:,int(x[t])])
    
        return y, x

    def forward(self, y):
        T = len(y)
        
        # Forward Pass

        # Python indexes starting from zero so
        # log \alpha_{k|k} will be in log_alpha[:,k-1]
        # log \alpha_{k|k-1} will be in log_alpha_pred[:,k-1]
        log_alpha  = np.zeros((self.S, T))
        log_alpha_pred = np.zeros((self.S, T))
        for k in range(T):
            if k==0:
                log_alpha_pred[:,0] = self.logpi
            else:
                log_alpha_pred[:,k] = self.predict(log_alpha[:,k-1])

            log_alpha[:,k] = self.update(y[k], log_alpha_pred[:,k])
            
        return log_alpha, log_alpha_pred
            
    def backward(self, y):
        # Backward Pass
        T = len(y)
        log_beta  = np.zeros((self.S, T))
        log_beta_post = np.zeros((self.S, T))

        for k in range(T-1,-1,-1):
            if k==T-1:
                log_beta_post[:,k] = np.zeros(self.S)
            else:
                log_beta_post[:,k] = self.postdict(log_beta[:,k+1])

            log_beta[:,k] = self.update(y[k], log_beta_post[:,k])

        return log_beta, log_beta_post
        
    def forward_backward_smoother(self, y):
        log_alpha, log_alpha_pred = self.forward(y)
        log_beta, log_beta_post = self.backward(y)
        
        log_gamma = log_alpha + log_beta_post
        return log_gamma
        
    def correction_smoother(self, y):
        # Correction Smoother

        log_alpha, log_alpha_pred = self.forward(y)
        T = len(y)
        
        # For numerical stability, we calculate everything in the log domain
        log_gamma_corr = np.zeros_like(log_alpha)
        log_gamma_corr[:,T-1] = log_alpha[:,T-1]

        C2 = np.zeros((self.S, self.S))
        C3 = np.zeros((self.R, self.S))
        C3[y[-1],:] = normalize_exp(log_alpha[:,T-1], axis=None)
        for k in range(T-2,-1,-1):
            log_old_pairwise_marginal = log_alpha[:,k].reshape(1,self.S) + self.logA 
            log_old_marginal = self.predict(log_alpha[:,k])
            log_new_pairwise_marginal = log_old_pairwise_marginal + log_gamma_corr[:,k+1].reshape(self.S,1) - log_old_marginal.reshape(self.S,1)
            log_gamma_corr[:,k] = log_sum_exp(log_new_pairwise_marginal, axis=0).reshape(self.S)
            C2 += normalize_exp(log_new_pairwise_marginal, axis=None)
            C3[y[k],:] += normalize_exp(log_gamma_corr[:,k], axis=None)
        C1 = normalize_exp(log_gamma_corr[:,0])
        return C1, C2, C3, log_gamma_corr
    
    def forward_only_SS(self, y, V=None):
        # Forward only estimation of expected sufficient statistics
        T = len(y)
        
        if V is None:
            V1  = np.eye((self.S))
            V2  = np.zeros((self.S,self.S,self.S)) # s(a,b|x_k)
            V3  = np.zeros((self.R,self.S,self.S))
        else:
            V1, V2, V3 = V
            
        I_S1S = np.eye(self.S).reshape((self.S,1,self.S))
        I_RR = np.eye(self.R)
        
        for k in range(T):
            if k==0:
                log_alpha_pred = self.logpi
            else:
                log_alpha_pred = self.predict(log_alpha)

            if k>0:
                # Calculate p(x_{k-1}|y_{1:k-1}, x_k) 
                lp = np.log(normalize_exp(log_alpha)).reshape(self.S,1) + self.logA.T    
                P = normalize_exp(lp, axis=0)

                # Update
                V1 = np.dot(V1, P)             
                V2 = np.dot(V2, P) + I_S1S*P.reshape((1,self.S,self.S))    
                V3 = np.dot(V3, P) + I_RR[:,y[k-1]].reshape((self.R,1,1))*P.reshape((1,self.S,self.S))    

            log_alpha = self.update(y[k], log_alpha_pred)    
            p_xT = normalize_exp(log_alpha)    

        C1 = np.dot(V1, p_xT.reshape(self.S,1))
        C2 = np.dot(V2, p_xT.reshape(1,self.S,1)).reshape((self.S,self.S))
        C3 = np.dot(V3, p_xT.reshape(1,self.S,1)).reshape((self.R,self.S))
        C3[y[-1],:] +=  p_xT
        
        ll = log_sum_exp(log_alpha)
        
        return C1, C2, C3, ll, (V1, V2, V3)

    
    def train_EM(self, y, EPOCH=10, method='forward_only'):
        LL = np.zeros(EPOCH)
        params = []
        for e in range(EPOCH):
            params.append([self.pi,self.A,self.B])
            if method is 'correction_smoother':
                C1, C2, C3, log_gamma_corr = self.correction_smoother(y)
                ll = log_sum_exp(log_gamma_corr[:,0])
            elif method is 'forward_only':
                C1, C2, C3, ll, V = self.forward_only_SS(y)
            else:
                return
            LL[e] = ll
            p = normalize(C1 + 1e-15, axis=0).reshape(self.S)
            # print(p,np.size(p))            
            A = normalize(C2, axis=0)
            # print(A)
            B = normalize(C3, axis=0)
            # print(B)
            self.__init__(p, A, B)
            
        return LL, params
    
    
    def online_em(self, y, V=None, n_min=100, gamma=0.2,log_interval=1e4, learn_rate=-0.6, update_freq=1):
        T = len(y)
        LL = np.zeros(T)
        params = []
        
        if V is None:
            V1  = np.eye((self.S))
            V2  = np.zeros((self.S,self.S,self.S)) # s(x_k,x_{k-1}|x_k)
            V3  = np.zeros((self.R,self.S,self.S))
        else:
            V1, V2, V3 = V
            
        I_S1S = np.eye(self.S).reshape((self.S,1,self.S))
        I_RR = np.eye(self.R)
        
        for k in range(T):
            # save model params
            if np.mod(k,log_interval)==0: 
                params.append([self.pi,self.A,self.B])
                # log_gamma = self.forward_backward_smoother(y)
            # E step
            if k==0:
                log_alpha_pred = self.logpi
            else:
                log_alpha_pred = self.predict(log_alpha)

            if k>0:
                # Calculate p(x_{k-1}|y_{1:k-1}, x_k) 
                lp = np.log(normalize_exp(log_alpha)).reshape(self.S,1) + self.logA.T    
                P = normalize_exp(lp, axis=0)

                # Update
                V1 = np.dot(V1, P)             
                V2 = (1-gamma)*np.dot(V2, P) + gamma*I_S1S*P.reshape((1,self.S,self.S))    
                V3 = (1-gamma)*np.dot(V3, P) + gamma*I_RR[:,y[k-1]].reshape((self.R,1,1))*P.reshape((1,self.S,self.S))    
                
                
            log_alpha = self.update(y[k], log_alpha_pred)    
                
            p_xT = normalize_exp(log_alpha)    
        
            LL[k] = log_sum_exp(log_alpha)
            
            # M step
            if k > n_min:
                C1 = np.dot(V1, p_xT.reshape(self.S,1))
                C2 = np.dot(V2, p_xT.reshape(1,self.S,1)).reshape((self.S,self.S))
                C3 = np.dot(V3, p_xT.reshape(1,self.S,1)).reshape((self.R,self.S))
                C3[y[k],:] +=  p_xT
                
                p = normalize(C1 + 0.1, axis=0).reshape(self.S)
                A = normalize(C2, axis=0)
                B = normalize(C3, axis=0)
                self.__init__(p, A, B)
            if np.mod(k,update_freq)==0:
                gamma = np.power(k,learn_rate)
    
        return LL,params

    
    def forward_only_SS_c(self, y, log_alpha=None, V=None):
        # Forward only estimation of expected sufficient statistics
        T = len(y)
        
        if V is None:
            V1  = np.eye((self.S))
            V2  = np.zeros((self.S,self.S,self.S))
            V3  = np.zeros((self.R,self.S,self.S))
        else:
            V1, V2, V3 = V
            
        I_S1S = np.eye(self.S).reshape((self.S,1,self.S))
        I_RR = np.eye(self.R)
        
        for k in range(T):
            if log_alpha is None:
                log_alpha_pred = self.logpi
            else:
                log_alpha_pred = self.predict(log_alpha)
                
            # Update
            if k > 0:
                # Calculate p(x_{k-1}|y_{1:k-1}, x_k) 
                lp = np.log(normalize_exp(log_alpha)).reshape(self.S,1) + self.logA.T    
                P = normalize_exp(lp, axis=0)
                
                V1 = np.dot(V1, P)             
                V2 = np.dot(V2, P) + I_S1S*P.reshape((1,self.S,self.S)) 
                V3 = np.dot(V3, P) + I_RR[:,y[k-1]].reshape((self.R,1,1))*P.reshape((1,self.S,self.S))   

            log_alpha = self.update(y[k], log_alpha_pred)    
            p_xT = normalize_exp(log_alpha)    

        C1 = np.dot(V1, p_xT.reshape(self.S,1))
        C2 = np.dot(V2, p_xT.reshape(1,self.S,1)).reshape((self.S,self.S))
        C3 = np.dot(V3, p_xT.reshape(1,self.S,1)).reshape((self.R,self.S))
        C3[y[-1],:] +=  p_xT
        
        return C1, C2, C3, V1, V2, V3, log_alpha
    
    
    
    def online_em_c(self, y, V=None, n_min=100, gamma=0.2,log_interval=1e4, learn_rate=-0.80):
        T = len(y)
        LL = np.zeros(T)
        params = []
        
        C1  = np.zeros(self.S)
        C2  = np.zeros((self.S,self.S))
        C3  = np.zeros((self.R,self.S))
            
        I_S1S = np.eye(self.S).reshape((self.S,1,self.S))
        I_RR = np.eye(self.R)
        
        for k in range(T):
            # save model params
            if np.mod(k,log_interval)==0: 
                params.append([self.pi,self.A,self.B])
            # E step
            if k==0:
                log_alpha_pred = self.logpi
                log_alpha = self.update(0, log_alpha_pred)  
            else:
                # calculate sufficient states
                C1_fb, C2_fb, C3_fb, V1_fb, V2_fb, V3_fb, log_alpha = self.forward_only_SS_c(y[k],log_alpha)
                C1 = (1-gamma)*C1 + gamma*C1_fb.reshape(self.S)            
                C2 = (1-gamma)*C2 + gamma*C2_fb   
                C3 = (1-gamma)*C3 + gamma*C3_fb  
                
            # M step
            if k > n_min:
                p = normalize(C1 + 0.1, axis=0)
                A = normalize(C2, axis=0)
                B = normalize(C3, axis=0)
                self.__init__(p, A, B)
                gamma = np.min((0.2,np.power(k,learn_rate)))
    
        return LL,params

hmm = HMM.from_random_parameters(S=3,R=10)

L = 100

y,x = hmm.generate_sequence(L)
log_gamma = hmm.forward_backward_smoother(y)

print("LL of the generative model is {:.4f}".format(log_sum_exp(log_gamma[:,0])[0]))

print("Results with the Forward Smoother")
C1, C2, C3, ll, V = hmm.forward_only_SS(y)

print(C1)
print(np.sum(C1))
print("\n")

print(C2)
print(np.sum(C2))
print("\n")

print(C3)
print(np.sum(C3))
print("\n")

print("Results with the Correction Smoother")
C1_corr, C2_corr, C3_corr, lg = hmm.correction_smoother(y)

print(C1_corr)
print(np.sum(C1_corr))
print("\n")

print(C2_corr)
print(np.sum(C2_corr))
print("\n")

print(C3_corr)
print(np.sum(C3_corr))
print("\n")

hmm = HMM.from_random_parameters(S=3,R=10)
y,x = hmm.generate_sequence(100)

LL, params = hmm.train_EM(y, 100)
print(LL)

