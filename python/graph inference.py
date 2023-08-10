import numpy as np

from sklearn.covariance import GraphLasso, GraphLassoCV

import numpy as np
def sparseinverse_cov(d):
    """Generate a covariance matrix with Sparse Inverse method.
    Sparse graph where each element depends only on a subset of others."""
    # W is a weight symmetric matrix.
    # It has -1 if two vertices are connected, 0 otherwise.
    # The diagonal is the sum of rows (or columns).
    W = np.zeros((d, d))
    for i in range(d):
        W[i, i+1:] = np.random.randint(-1, 1, d-i-1)
    W += W.T
    w = [abs(np.sum(vec)) for vec in W]
    laplacian = W + np.diag(w)
    return np.linalg.inv(laplacian + np.random.randn() ** 2 * np.eye(d))

theta_t = sparseinverse_cov(3)

theta_t

from sklearn.covariance import GraphLasso, GraphLassoCV

def random_covariance(n):
    Sinv = np.eye(n)
    idx = np.random.randint(2, size=n*n).reshape(n,n);
    for i in range(n):
        for j in range(n):
            Sinv[i,j] = idx[i,j]
    Sinv = Sinv + Sinv.T
    if np.min(np.linalg.eigh(Sinv)[0]) < 0:
        Sinv = Sinv + 1.1*np.abs(np.min(np.linalg.eigh(Sinv)[0]))*np.eye(n);

    S = np.linalg.inv(Sinv);
    return S, Sinv

n_dim = 10
n_samples = 50
alpha=2*np.sqrt(np.log(n_dim) / n_samples)
print "alpha:", alpha

# true_covariance = sparseinverse_cov(n_dim)
# true_covariance /= np.diag(true_covariance)[0]
true_covariance, Sinv = random_covariance(n_dim)
X = np.random.multivariate_normal(np.zeros(n_dim), true_covariance, n_samples)

gl = GraphLasso(mode='cd', alpha=alpha, verbose=0, max_iter=200)
gl = GraphLassoCV(verbose=0, max_iter=500, alphas=20)
gl.fit(X)

from rgi import admm_covariance; reload(admm_covariance)

Cov, Z, hist = admm_covariance.covsel(X, .05,rho=2,alpha=alpha, verbose=0)

# print gl.error_norm(true_covariance)
# print np.linalg.norm(gl.covariance_ - true_covariance)

print(np.linalg.norm(Sinv - Cov))

print(np.linalg.norm(gl.precision_ - Sinv))

Z

np.set_printoptions(precision=1, suppress=True)

gl.precision_

Sinv

Cov

import pandas as pd
import numpy as np
from sklearn.covariance import GraphLasso, GraphLassoCV, empirical_covariance

df_x = pd.read_csv("/home/fede/projects_local/kdvs/", delimiter='\t', comment='#')
df_y = pd.read_csv("/home/fede/projects_local/petretto/data/fmf_labels.txt", delimiter='\t')

X = df_x.values
y = df_y.values.ravel()

df_x.loc[:, df_x.columns.str.startswith("FMF_C")].values

emp_cov = empirical_covariance(X[y=='A'], assume_centered=False)

gl = GraphLasso(verbose=1, max_iter=200)
gl.fit(df_x.loc[:, df_x.columns.str.startswith("FMF_C")].values.T)



n, d = 400, 10

X = np.random.randn(n, d)
beta = np.ones(d); beta[:d//2] = 0

y = X.dot(beta) + np.random.randn(n) * 0.1

beta

from sklearn.linear_model import Lasso

beta_lasso = Lasso(alpha=0.1).fit(X, y).coef_

beta_lasso

from rgi import admm_lasso as lasso; reload(lasso)

z = lasso.lasso(X, y, lamda=0.1, rho=1, alpha=1)

z



from rgi import admm_group_lasso as grouplasso; reload(grouplasso)

z_group = grouplasso.group_lasso(X, y, lamda=0.1, p=[[i] for i in range(10)],
                                 rho=1, alpha=1)

z_group

print "Error skle: %.3f" % np.linalg.norm(beta_lasso - beta)
print "Error admm: %.3f" % np.linalg.norm(z - beta)
print "Error admm_group: %.3f" % np.linalg.norm(z_group - beta)



from rgi import admm_group_lasso_overlap as grouplassooverlap; reload(grouplassooverlap)

z_group = grouplassooverlap.GroupLassoOverlap(
        alpha=2, rho=2, groups=[[0,1,2,3,4], [3,4,5], [5,6,7,8,9]], verbose=0, max_iter=1000, tol=1e-6).fit(
X, y).coef_

print "Error admm_group_overlap: %.3f" % np.linalg.norm(z_group - beta)

get_ipython().magic('load_ext rpy2.ipython')

get_ipython().run_cell_magic('R', '', '# install.packages("psych")\n# install.packages("matrixcalc")\n# install.packages("corpcor")\n# install.packages("Matrix")\n# install.packages("glasso")\ninstall.packages("qgraph")')

get_ipython().run_cell_magic('R', '', '\nlibrary(psych)\nlibrary(matrixcalc)\nlibrary(Matrix)\nlibrary(glasso)\nlibrary(qgraph)\n# E-step in the optimization algorithm:\nEstep <- function(\n  S, # Sample covariance\n  Kcur, # Current estimate for K\n  obs # Logical indicating observed\n)\n{\n  stopifnot(require("Matrix"))\n  stopifnot(isSymmetric(S))\n  if (missing(Kcur))\n  {\n    if (missing(obs)) \n    {\n      Kcur <- diag(nrow(S)) \n    } else \n    {\n      Kcur <- diag(length(obs))\n    }\n  }\n  stopifnot(isSymmetric(Kcur))\n  if (missing(obs)) obs <- 1:nrow(Kcur) %in% 1:nrow(S)\n  \n  # To make life easier:\n  H <- !obs\n  O <- obs\n  \n  \n  # Current estimate of S:\n  library(corpcor)\n  Scur <- pseudoinverse(Kcur)\n  \n  # Expected Sigma_OH:\n  Sigma_OH <- S %*% pseudoinverse(Scur[O,O]) %*% Scur[O, H]\n  #   Sigma_OH\n  \n  # Expected Sigma_H:\n  Sigma_H <- Scur[H, H] - Scur[H,O] %*% pseudoinverse(Scur[O,O]) %*% Scur[O,H] + Scur[H,O] %*% pseudoinverse(Scur[O,O]) %*% S %*% pseudoinverse(Scur[O,O]) %*% Scur[O, H]\n  \n  # Construct expected sigma:\n  Sigma_Exp <- rbind(cbind(S,Sigma_OH),cbind(t(Sigma_OH), Sigma_H))  \n  \n  return(Sigma_Exp)\n}\n\n# M-step in the optimization algorithm:\nMstep <- function(\n  Sexp, # Expected full S\n  obs, # Logica indiating oberved variables\n  rho = 0,\n  lambda\n)\n{\n  if (!is.positive.definite(Sexp))\n  {\n    Sexp <- as.matrix(nearPD(Sexp)$mat)\n    warning("Expected covariance matrix is not positive definite")\n  }\n  # Rho matrix:\n  n <- nrow(Sexp)\n  RhoMat <- matrix(rho, n, n)\n  RhoMat[!obs,] <- 0\n  RhoMat[,!obs] <- 0\n  \n  # Lambda:\n  zeroes <- which(!lambda, arr.ind = TRUE)\n  zeroes[,2] <- which(!obs)[zeroes[,2]]\n  \n  if (nrow(zeroes) > 0){\n    K <- glasso(Sexp, RhoMat, penalize.diagonal=FALSE, zero = zeroes)$wi \n  } else {\n    K <- glasso(Sexp, RhoMat, penalize.diagonal=FALSE)$wi\n  }\n  \n  return(K)  \n}\n\n### Main lvglasso function\nlvglasso <- function(\n  S, # Sample cov\n  nLatents, # Number of latents\n  rho = 0, # Penalty\n  thr = 1.0e-4, # Threshold for convergence (sum absolute diff)\n  maxit = 1e4, # Maximum number of iterations\n  lambda, # Logical matrix indicating the free factor loadings. Defaults to full TRUE matrix.\n  ... # qgraph arguments\n)\n{\n  if (missing(nLatents)){\n    stop("\'nLatents\' must be specified")\n  }\n  \n  nobs <- nrow(S)\n  ntot <- nobs + nLatents\n  \n  if (missing(lambda) || is.null(lambda)){\n    lambda <- matrix(TRUE, nobs, nLatents)\n  }\n  \n  if (nrow(lambda) != nobs | ncol(lambda) != nLatents) stop("Dimensions of \'lambda\' are wrong.")\n  \n  # PCA prior for K:\n  efaRes <- principal(S, nfactors = nLatents)\n  \n#   # If sampleSizeis missing, set to 1000. Is only used for prior anyway.\n# #   if (missing(sampleSize)){\n#     sampleSize <- 1000\n# #   }\n# \n#   #   Get prior for K:\n#   efaRes <- fa(S, n.obs= sampleSize, nfactors=nLatents)\n  resid <- residuals(efaRes)\n  class(resid) <- "matrix"\n  load <- loadings(efaRes)\n  class(load) <- "matrix"\n  r <- efaRes$r.scores\n  class(r) <- "matrix"\n  r <- diag(diag(r))\n#   \n#   # Stupid nonanalytic way to get prior:\n#   # Simulate N random variales:\n#   #   eta <- rmvnorm(10000, rep(0, nLatents), r)\n#   # \n#   #   # Simulate oserved scores:\n#   #   Y <- eta %*% t(load) + rmvnorm(10000, rep(0, nobs), diag(diag(resid)\n  # RAM FRAMEWORK:\n  \n  Sym <- rbind(cbind(diag( efaRes$uniquenesses),matrix(0,nobs,nLatents) ),cbind(t(matrix(0,nobs,nLatents) ),r))\n  As <- matrix(0, ntot, ntot)\n  if (nLatents > 0) As[1:nobs, (nobs+1):ntot] <- load\n  \n  Sigma <- solve(diag(ntot) - As) %*% Sym %*% t(solve(diag(ntot) - As))\n  \n  # Compute K:\n# browser()\n  K <- solve(cor2cov(cov2cor(Sigma),c(sqrt(diag(S)), rep(1, nLatents) )))\n  #  K <- K  \n#   K <- cov2cor(K)\n      K=round(K,10)\n  if (!is.positive.definite(K))\n  {\n    K <- as.matrix(nearPD(K)$mat)\n#     warning("Expected covariance matrix is not positive definite")\n  }\n\n# browser()\n\n  #   K <- matrix(-0.5,ntot,ntot)\n  #   K[1:nobs,1:nobs] <- 0\n  #   diag(K) <- 1\n  \n  #   K <- round(K,2)\n  \n  #   K <- EBICglasso(cov(cbind(Y,eta)), sampleSize)\n  #   K <- as.matrix(forceSymmetric(cbind(rbind(pseudoinverse(resid),t(-load)), rbind(-load,pseudoinverse(r)))))\n  \n  # K <- as.matrix(forceSymmetric(cbind(rbind(pseudoinverse(resid),-0.5), rbind(rep(-0.5,nobs),1))))\n  #   K <- as.matrix(forceSymmetric(cbind(rbind(diag(nrow(S)),t(-load)), rbind(-load,pseudoinverse(r)))))\n  # browser()\n  #   K <- matrix(-0.5,ntot,ntot)\n  #   K[1:nobs,1:nobs] <- 0\n  #   diag(K) <- 1\n  #   obs <- c(rep(TRUE,nrow(S)), rep(FALSE,nLatents))\n  # \n  #   K <- rbind(cbind(2*diag(nobs),-load),cbind(t(-load), 2*diag(ntot - nobs)))\n  # rownames(K) <- colnames(K) <- NULL\n  # \n  # diag(K) <- diag(K) - min(eigen(pseudoinverse(K))$values)\n  \n  # #   K[1:nobs,1:nobs] <- 0\n  #   diag(K) <- 1\n  obs <- c(rep(TRUE,nrow(S)), rep(FALSE,nLatents))\n# \n# # Stupid prior:\n# K <- matrix(0, ntot, ntot)\n# diag(K) <- 1\n# if (nLatents > 0) K[1:nobs, (nobs+1):ntot] <- K[ (nobs+1):ntot, 1:nobs] <- -1/nobs\n  \n  #   is.positive.definite(as.matrix(forceSymmetric(Estep(S, K, obs))))\n  ### EM ###\nit <- 1\nKold <- K\n  repeat\n  {\n    Sexp <- as.matrix(forceSymmetric(Estep(S, K, obs)))\n    #     Sexp <- cor2cov(cov2cor(Sexp),ifelse(obs,sqrt(diag(Sexp)),1))\n    K <- as.matrix(forceSymmetric(Mstep(Sexp, obs, rho, lambda)))\n    #     qgraph(wi2net(K), layout = "spring")\n    \n    # Check for convergence:\n    if (sum(abs(cov2cor(pseudoinverse(Kold)[obs,obs]) - cov2cor(pseudoinverse(K)[obs,obs]))) < thr){\n      break\n    } else {\n      it <- it + 1\n      if (it > maxit){\n        warning("Algorithm did not converge!")\n        break\n      } else {\n        Kold <- K\n      }\n    }\n  }\n\n  if (!is.null(colnames(S)))\n  {\n    colnames(K) <- c(colnames(S),rep(paste0("F",seq_len(nLatents)))) \n  }\n  \n  if (is.null(rownames(S)))\n  {\n#       print(c(rep(paste0("N",seq_len(dim(K)[1] - nLatents))),rep(paste0("F",seq_len(nLatents)))) )\n    rownames(K) <-c(rep(paste0("N",seq_len(dim(K)[1] - nLatents))),rep(paste0("F",seq_len(nLatents)))) \n  }\n\n  # Partial correlations:\n  pc <- qgraph:::wi2net(K)\n  diag(pc) <- 1\n  \nrownames(pc) <- colnames(pc) <- rownames(K) <- colnames(K)\n\n# Compute psychometric matrices:\nTheta <- solve(K[obs, obs])\nLambda <- -Theta %*% K[obs,!obs]\nPsi <- solve(K[!obs, !obs] - t(Lambda) %*% K[obs, obs] %*% Lambda)\n\n# Return list mimics glasso:\nRes <- list(\n  w = pseudoinverse(K), # Estimated covariance matrix\n  wi = K, # Estimated precision matrix\n  pcor = pc, # Estimated partial correlation matrix\n  observed = obs, # observed and latents indicator\n  niter = it, # Number of iterations used in the algorithm\n  lambda = Lambda,\n  theta = Theta,\n  psi = Psi\n    )\n\n  class(Res) <- "lvglasso"\n\n  return(Res)\n}')

from sklearn.covariance import empirical_covariance

covariance = empirical_covariance(X)

get_ipython().run_cell_magic('R', '-i covariance -o Res', '\nRes <- lvglasso(covariance, 3)')



