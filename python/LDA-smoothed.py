import sys,getopt
from numpy import array,matrix,diag
from scipy import sum,log,exp,mean,dot,ones,zeros
from scipy.special import polygamma, gamma
from scipy.linalg import norm
from random import random
import numpy as np
import os
import pandas as pd

os.chdir("/home/3928941380/Downloads")

def main(emmax = 100, beta_estimate = 1):
    # set parameters
    k = 10 # of classes to assume
    #emmax = 2 # of maximum VB-EM iteration (default 100)
    demmax = 20 # of maximum VB-EM iteration for a document
    epsilon = 0.0001 # A threshold to determine the whole convergence of the estimation
    
    # Train
    train = open("train.txt",'r').read()
    alpha,phi, beta = ldamain(train, k, beta_estimate, emmax, demmax, epsilon)
    
    # Write
    writer = open('output-alpha.txt','w')
    writer.write(str(alpha.tolist()))
    writer.close() 
    
    writer = open('output-phi.txt','w')
    writer.write(str(phi.tolist()))
    writer.close()
    
    return alpha, phi, beta

def ldamain(train, k, beta_estimate, emmax=100, demmax=20, epsilon=1.0e-4):
    d = [ zip(*[ [int(x) for x in w.split(':')] for w in L.split()]) for L in train.split('\n') if L ]
    
    data = []
    for L in train.split("\n"):
        if L == "":
            continue

        id_ = [int(w.split(":")[0]) for w in L.split(" ")]
        w_count = [int(w.split(":")[1]) for w in L.split(" ")]

        data.append([id_, w_count])
    
    return lda.train(data,k,beta_estimate,emmax,demmax, epsilon)

class lda():
    '''
    Latent Dirichlet Allocation, standard model.
    [alpha,phi] = lda.train(d,k,[emmax,demmax])
    d      : data of documents
    k      : # of classes to assume
    emmax  : # of maximum VB-EM iteration (default 100)
    demmax : # of maximum VB-EM iteration for a document (default 20)
    '''
    
    @staticmethod
    def train(d, k, beta_estimate, emmax=100, demmax=20, epsilon=1.0e-4):
        '''
        Latent Dirichlet Allocation, standard model.
        [alpha,phi] = lda.train(d,k,[emmax,demmax])
        d      : data of documents
        k      : # of classes to assume
        emmax  : # of maximum VB-EM iteration (default 100)
        demmax : # of maximum VB-EM iteration for a document (default 20)
        '''
        
        # # of documents
        M = len(d)
        # # of words
        L = max(map(lambda x: max(x[0]), d)) + 1
        
        # initialize
        beta = matrix(np.full((L, 1), 100, dtype=float) / L) # k x 1 matrix
        phi = matrix(ones((k, L)) / L)
        alpha = matrix(lda.normalize(sorted([random() for i in range(k)], reverse=True))).T
        gammas = matrix(zeros((M, k)))
        lik = 0
        plik = lik
        n_kv = matrix(np.random.rand(k, L))
        phis = matrix(zeros((k, L)))
        
        #print ('number of documents (M)      = {0}'.format(M))
        #print ('number of words (l)          = {0}'.format(L))
        #print ('number of latent classes (k) = {0}'.format(k))
        
        for j in range(emmax):
            if j % 10 == 0:
                print ('iteration {0}/{1}..\t'.format(j+1, emmax))
            #vb-esstep
            
            nt_keep = []
            for i in range(M):
                gamma, q, nt, n_kv, xi_kv = lda.vbem(d[i], phi, alpha, beta, n_kv, demmax)
                nt_keep.append(nt)
                gammas[i,:] = gamma.T
                phis = lda.accum_phi(phis,q,d[i], xi_kv)
            #vb-mstep
            alpha = lda.fpi_alpha(alpha, nt_keep)
                    #alpha = lda.newton_alpha(gammas)
            if beta_estimate == 1:
                beta = lda.fpi_beta1(n_kv, beta)
            else:
                beta = lda.fpi_beta2(n_kv, beta)
            phi = lda.mnormalize(phis,1)
            #converge?
#             lik = lda.lda_lik(d, phi, gammas)
#             print ('log-likelihoood =', lik)
#             if j > 1 and abs((lik - plik) / lik) < epsilon:
#                 if j < 5:
#                     return lda.train(d, k, emmax, demmax) # try again
#                 print ('converged')
#                 return alpha, phi
#             plik = lik
            
        return alpha, phi, beta
                

    @staticmethod
    def vbem(di, phi, alpha0, beta, n_kv, emmax=20):
        '''
        calculates a document and words posterior for a document d.
        alpha  : Dirichlet posterior for a document d
        q      : (Nd * K) matrix of word posterior over latent classes
        di      : document data / here, only one sentence
        phi   : 
        alpha0 : Dirichlet prior of alpha
        emmax  : maximum # of VB-EM iteration.
        '''
        digamma = lambda x: polygamma(0,x)

        Nd = len(di[0])
        k = len(alpha0)
        q = zeros((Nd, k))
        nt = matrix(ones((1, k)) * Nd / k).T # initialize n_dk
        pnt = nt
        # xi_kv = n_kv + beta.T
        
        for j in range(emmax):
            #vb-estep
            q = matrix( matrix(exp(digamma((n_kv + beta.T)[:,di[0]])).T) * diag(exp(digamma(alpha0 + nt))[:,0]))
            q = lda.mnormalize(q , 1)
                    # q(z_d,i =k)
                    # alpha0 + ntでよくわからないけど、alphaをアップデートしてるからe-step?
                    # better to look at original C code
                       #  ap[k]というのが、式のexpの分子部分相当
                        #  それにphiをかけて最後に正規化している
            #vb-mstep
            nt =  q.T * matrix(di[1]).T
                    # nt:  probably expectation part in Sato Equation(3.89), same as bottom in Sato p.75
                    #  Sato p.75にあるように本来ならば文章中の一語ずつチェックするが、データセットには
                    #  各単語ごとの出現回数しかないので、単語が登場するたびに足すのではなく、単語の出現回数×qをして
                    #  足しあわせている
                    # better to look at original C code
            for k_index in range(k): 
                n_kv[k_index, di[0]]  = q.T[k_index, :]
                # Sato p.77下の更新式だけど、ここでは全てのdに対しては更新しないで、vbem()に流れてきているものだけを更新 <-- ！！！！！！この方法で良いのか不明！！！！！

            #converge?
            if j > 1 and np.absolute((nt - pnt).sum()) / k < 3 :
                break
            pnt = nt.copy()
            pnkv = n_kv.copy()

        alpha = alpha0 + nt # corresponds to Sato Eq (3.89) / dの全てのkに対して一度にしている
                                        # nt:  probably expectation part in Sato Equation(3.89)
        xi_kv = n_kv + beta.T  # Sato (3.95)
        return alpha, q, nt, n_kv, xi_kv
    
    
    @staticmethod
    def fpi_alpha(alpha, nt_keep):
        # fixed point iteration for alpha / Sato p.112
        K = alpha.shape[0]
        M = len(nt_keep)
        digamma = lambda x: polygamma(0,x)

        for k in range(K):
            alpha_k = alpha[k, 0]

            numerator = 0 ; denominator = 0
            for m in range(M):
                nt = nt_keep[m]
                nd = nt.sum() # Sato p.114

                # numerator
                numerator += (digamma(nt[k,0] + alpha_k) - digamma(alpha_k)) * alpha_k

                # denominator
                denominator += digamma(nd + alpha.sum()) - digamma(alpha.sum())

            alpha[k,0] = numerator / denominator

        return alpha
    
    @staticmethod
    def fpi_beta1(n_kv, beta):
        K, V = n_kv.shape
        new_beta = matrix(zeros((V, 1)))
        digamma = lambda x: polygamma(0,x)

        for v in range(V):
            numerator = 0 ; denominator = 0
            for k in range(K):
                numerator += (digamma(n_kv[k, v] + beta[v, 0]) - digamma(beta[v, 0])) * beta[v, 0]
                denominator += (digamma(n_kv[k, :].sum() + beta[v,0]) - digamma(beta.sum()))

            new_beta[v, :]  = numerator / denominator
            
        return lda.normalize(new_beta)
    
    @staticmethod
    def fpi_beta2(n_kv, beta): # Sato (3.194)
        K, V = n_kv.shape
        new_beta = matrix(zeros((V, 1)))
        digamma = lambda x: polygamma(0,x) 

        component1 = 0
        for k in range(K):
            for v in range(V):
                component1 += n_kv[k,v]

        numerator = 0
        for v in range(V):
            for k in range(K):
                numerator += (digamma(n_kv[k,v] + beta[0,0]) - digamma(beta[0,0])) * beta[0,0]

        denominator = 0
        for k in range(K):
            denominator += digamma(component1 + beta[0,0]) - digamma(V * beta[0,0])

        new_beta.fill(numerator / denominator / V)
        return lda.normalize(new_beta)

    @staticmethod
    def accum_phi(phis, q, di, xi_kv):
        '''
        phis = accum_phi(phis,q,t)
        accumulates word posteriors to latent classes.
        phis : (V * K) matrix of summand
        q     : (L * K) matrix of word posteriors
        t     : document of struct array
        '''
        new_phis  = matrix(zeros((phis.shape[0], phis.shape[1])))
        #for k in range(phis.shape[0]):
        for k in range(phis.shape[0]):
            new_phis[k, :] = matrix(np.random.dirichlet(np.squeeze(np.asarray(xi_kv[k,:])) , 1))

            #new_phis[k, :] =  lda.normalize(new_phis[k, :])
        
        return new_phis
    
    @staticmethod
    def lda_lik(d, phi, gammas):
        '''
        lik = lda_lik(d, phi, gammas)
        returns the likelihood of d, given LDA model of (phi, gammas).
        '''
        egamma = matrix(lda.mnormalize(gammas, 1))
        lik = 0
        M = len(d)
        for i in range(M):
            t = d[i]
            lik += (matrix(t[1]) * log(matrix(phi[:,t[0]]).T * egamma[i,:].T))[0,0]
        return lik
    
    @staticmethod
    def normalize(v):
        return v / sum(v)
    
    @staticmethod
    def mnormalize(m, d=0):
        '''
        x = mnormalize(m, d)
        normalizes a 2-D matrix m along the dimension d.
        m : matrix
        d : dimension to normalize (default 0)
        '''
        m = array(m)
        v = sum(m, d)
        if d == 0:
            return m * matrix(diag(1.0 / v))
        else:
            return matrix(diag(1.0 / v)) * m
        
    @staticmethod
    def converged(u, udash, threshold=1.0e-3):
        '''
        converged(u,udash,threshold)
        Returns 1 if u and udash are not different by the ratio threshold
        '''
        return norm(u - udash) / norm(u) < threshold

if __name__ == '__main__':
    alpha, phi, beta = main(emmax=100, beta_estimate = 0)
    
    res_b = pd.DataFrame(phi)
    for i in range(phi.shape[0]):
        res_temp = res_b.ix[i, :].sort_values(ascending=False)[:10]
        res_temp.index += 1
        print("Topic: ", i)
        print(res_temp)
        print("\n")

phi.shape

alpha

beta

alpha, phi, beta = main(emmax=100, beta_estimate = 1)

res_b = pd.DataFrame(phi)
for i in range(phi.shape[0]):
    res_temp = res_b.ix[i, :].sort_values(ascending=False)[:10]
    res_temp.index += 1
    print("Topic: ", i)
    print(res_temp)
    print("\n")

