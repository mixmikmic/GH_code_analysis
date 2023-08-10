import ipystata

from IPython.display import IFrame

get_ipython().run_cell_magic('stata', '', '\nclear all\nset more off \nuse http://fmwww.bc.edu/repec/bocode/t/traindata.dta\nset seed 90210')

get_ipython().run_cell_magic('stata', '', '\ncapture ssc install mixlogit')

get_ipython().run_cell_magic('stata', '', '\nmixlogit y, rand(price contract local wknown tod seasonal) group(gid) id(pid)')

get_ipython().run_cell_magic('stata', '--mata', "\nreal matrix drawb_betaW(beta, W) {\n    return( mean(beta) + rnormal(1, cols(beta) ,0 ,1 ) * cholesky(W)' )\n}")

get_ipython().run_cell_magic('stata', '--mata', "\nreal matrix drawW_bbeta(beta, b) {\n    v  = rnormal( cols(b) + rows(beta), cols(b), 0, 1)\n    S1 = variance(beta :- b)\n    S  = invsym((cols(b)*I(cols(b)) + rows(beta) * S1)/(cols(b) + rows(beta)))\n    L  = cholesky(S)\n    R = (L*v')*(L*v')' / (cols(b) + rows(beta))\n    return(invsym(R))\n}")

get_ipython().run_cell_magic('stata', '--mata', "\nreal scalar lncp(real rowvector beta_rn,\n                 real rowvector b,\n                 real matrix Winv,\n                 real matrix ldetW,\n                 real matrix y,\n                 real matrix Xr,\n                 real matrix cid)\n{\n    \n    real scalar i, lnp, lnprior\n    real matrix z, Xrp, yp, mus\n \n    z   = panelsetup(cid, 1)\n    \n    lnp = 0\n    \n    for (i=1; i<=rows(z); i++) {\n        Xrp  = panelsubmatrix(Xr, i, z)\n         yp  = panelsubmatrix(y,  i, z)\n         mus = rowsum(Xrp:*beta_rn)\n         max = max(mus)\n         sum = max + ln(colsum(exp(mus :- max)))\n         lnp = lnp + colsum(yp:*mus) :- sum\n    }\n\n    lnprior= -1/2*(beta_rn - b)*Winv*(beta_rn - b)' - 1/2*ldetW - cols(b)/2*ln(2*pi())\n\n    return(lnp + lnprior)\n}")

IFrame('AlgPic.pdf', width=800, height=300)

get_ipython().run_cell_magic('stata', '--mata', '\nst_view(y=., .,   "y")\nst_view(X=., .,   "price contract local wknown tod seasonal")\nst_view(pid=., ., "pid")                                            // Individual identifier\nst_view(gid=., ., "gid")                                            // choice occasions')

get_ipython().run_cell_magic('stata', '--mata', "m = panelsetup(pid, 1)\n\nb = J(1, 6, 0)\nW = I(6)*6\nbeta = b :+ sqrt(diagonal(W))':*rnormal(rows(m), cols(b), 0, 1)")

get_ipython().run_cell_magic('stata', '--mata', '\nits    = 20000\nburn   = 10000\nnb     = cols(beta)\nbvals  = J(0, cols(beta), .)                    // Store draws of the mean parameter \nWvals  = J(0, cols(rowshape(W, 1)), .)          // Store draws of the variance parameters\n\npropVs = J(rows(m), 1, rowshape(W, 1))             \npropms = J(rows(m), 1, b)                          // Store the proposal mean and variance data for each individual\naccept = J(rows(m), 1, 0)                          // Store a count of acceptances so we can monitor acceptance rates\n\natarg  = .25                                       // target acceptance rate\nlam    = J(rows(m), 1, 2.38^2/nb)                  // initial value of the scaling parameter\ndamper = 1                                         // damping parameter value')

get_ipython().run_cell_magic('stata', '--mata', "\nfor (i=1; i<=its; i++) {\n    \n    b = drawb_betaW(beta, W/rows(m))\n    W = drawW_bbeta(beta, b)\n    \n    bvals = bvals \\ b\n    Wvals = Wvals \\ rowshape(W, 1)\n    \n    beta_old = beta\n    \n    Winv  = invsym(W)\n    ldetW = ln(det(W))\n    \n    for (j=1; j<=rows(m); j++) {\n        \n        yi   = panelsubmatrix(y, j,m)\n        Xi   = panelsubmatrix(X, j, m)\n        gidi = panelsubmatrix(gid, j, m)\n        \n        propV = rowshape(propVs[j, ], nb)\n    \n        beta_old = beta[j, ]\n        beta_hat = beta[j, ] + lam[j]*rnormal(1,nb,0,1)*cholesky(propV)' \n\n        old = lncp(beta_old, b, Winv, ldetW, yi, Xi, gidi)\n        pro = lncp(beta_hat, b, Winv, ldetW, yi, Xi, gidi)\n        \n        if  (pro == . )     alpha = 0\n        else if (pro > old) alpha = 1\n        else                alpha = exp(pro - old)\n        \n        if (runiform(1, 1) < alpha) {\n            beta[j, ] = beta_hat\n            accept[j] = accept[j] + 1\n        }\n\n        lam[j] = lam[j]*exp(1/(i+1)^damper*(alpha - atarg))\n        propms[j, ] = propms[j, ] + 1/(i + 1)^damper*(beta[j, ] - propms[j, ])\n        propV       = propV + 1/(i + 1)^damper*((beta[j, ] - propms[j, ])'(beta[j, ] - propms[j, ]) - propV)\n        _makesymmetric(propV)\n        propVs[j, ] = rowshape(propV, 1)    \n    }    \n}\n\narates = accept/its       ")

get_ipython().run_cell_magic('stata', '', 'preserve\nclear \ngetmata (b*) = bvals\n\nsum b*')

get_ipython().run_cell_magic('stata', '--graph', 'gen t = _n\ntsset t\nforvalues i=1/6 {\n    quietly tsline b`i\', saving(bg`i\'.gph, replace)\n\tlocal glist `glist\' "bg`i\'.gph"\n}\ngraph combine `glist\'')

get_ipython().run_cell_magic('stata', '--graph', 'clear \ngetmata arates lam\nhist arates')

get_ipython().run_cell_magic('stata', '--graph ', 'hist lam')



