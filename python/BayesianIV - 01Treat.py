import ipystata

get_ipython().run_cell_magic('stata', '', 'clear all \nuse http://www.stata-press.com/data/r13/union3\nset more off\nkeep if union != . \nkeep if tenure != .\ndescribe \nset seed 5150')

get_ipython().run_cell_magic('stata', '', 'reg wage age grade smsa black tenure if union == 1\nmat binit1 = e(b)\n\nreg wage age grade smsa black tenure if union == 0\nmat binit0 = e(b) \n\nprobit union south black tenure\nmat einit = e(b)')

get_ipython().run_cell_magic('stata', '--mata', '    st_view(y=., ., "wage")\n    st_view(tr=., ., "union")\n    st_view(X=., ., "age grade smsa black tenure")\n    st_view(W=., ., "south black tenure")\n    X = X, J(rows(y), 1, 1)\n    W = W, J(rows(y), 1, 1)\n\n    b1 = st_matrix("binit1")        \n    b0 = st_matrix("binit0")\n    e = st_matrix("einit")\n\n    nb = cols(b1)\n    ne = cols(e)\n\n    lnsd1 = 0                      \n    lnsd0 = 0\n    v10   = 0\n    v1t   = 0\n    v0t   = 0')

get_ipython().run_cell_magic('stata', '--mata', '    real matrix invnormstab(X) {\n        XHat = editvalue(X, 0, 1e-323)\n        XHat = editvalue(XHat, 1, 1e-16 )\n        return(XHat)\n    }\n    real scalar ln_L(real matrix errs, real matrix Sigma) {\n        part1 = -cols(errs)*rows(errs)/2*ln(2*pi())\n        part2 = -1/2*colsum(rowsum( (errs*invsym(Sigma):*errs)))\n        part3 = -rows(errs)/2*ln(det(Sigma))\n        return(part1 + part2 + part3)\n    }')

get_ipython().run_cell_magic('stata', '--mata', "    y0Hat = X*b0' + rnormal(rows(y),1,0,1)*exp(lnsd0)\n    y1Hat = X*b1' + rnormal(rows(y),1,0,1)*exp(lnsd1)\n    y1 = tr:*y + (1 :- tr):*y1Hat\n    y0 = (1 :- tr):*y + tr:*y0Hat\n\n    muz = W*e'\n    et  = invnormstab( normal(-muz) + (1 :- normal(-muz)):*runiform(rows(muz),1) )\n    ent = invnormstab( normal(-muz):*runiform(rows(muz),1) )\n    z = muz + et:*tr + ent:*(1 :- tr)\n    \n    m0 = X*b0'\n    m1 = X*b1'\n    mt = W*e'\n    ey1 = (y1 - m1)\n    ey0 = (y0 - m0)\n    et  = (z - mt)")

get_ipython().run_cell_magic('stata', '--mata', "    b1Hold        = b1\n    b0Hold        = b0\n    eHold         = e\n    sd1Hold       = lnsd1\n    sd0Hold       = lnsd0\n    v10Hold       = v10\n    v1tHold       = v1t\n    v0tHold       = v0t\n\n    draws = 1000\n    \n    XX = invsym(X'X)\n    WW = invsym(W'W)")

get_ipython().run_cell_magic('stata', '--mata', "\n    Sig12 = (v10, v1t)                                      //Compute Sigma[12] for y1\n    Sig22 = exp(lnsd0)^2, v0t \\ v0t, 1                      //Compute Sigma[22] for y1\n    Sig22m1 = invsym(Sig22)                                 //Invert Sigma[22]\n\n    CM = rowsum( (Sig12*Sig22m1):*(ey0, et) )               //This is Sigma[12]*S[22]^-1*[(y0,z)-(my0,mz)] in parallel\n    CV = exp(lnsd1)^2 - Sig12*Sig22m1*Sig12'                //variance of y1 conditional on y0, \n\n    mc1   = X*b1' + CM                                         //Mean of y1 conditional on y0,z\n    y1Hat = mc1 + rnormal(rows(y),1,0,1)*sqrt(CV)           //Drawing y1 from conditional distribution\n    y1    = tr:*y + (1 :-tr):*y1Hat                         //Replacing y1 with draws but only if y1 not observed\n\n    mb1 = XX*X'(y1 - CM)                                    //mean(b1) = (X'X)^{-1}(X'y1) y1 is purged of dependence on y0,z \n    vb1 = CV*XX                                             //Var(b1)\n    b1 = mb1 + cholesky(vb1)*rnormal(cols(b1), 1, 0, 1)     //Drawing new b1, given mean and variance\n    b1 = b1'                                                //row to column vector\n\n    m1 = X*b1'                                              //computation of new deviations from mean\n    ey1 = (y1 - m1)")

get_ipython().run_cell_magic('stata', '--mata', 'b1')

get_ipython().run_cell_magic('stata', '--mata', '    prosd1 = 1                                                                   //Standard dev. for proposal\n    gam   = 1                                                                       //Damping parameter\n    asta = .4                                                                    //Acceptance rate target\n    delta = 1\n\n    lnsd1Hat = lnsd1 + rnormal(1,1,0,1)*prosd1                                   //Draw from normal with mean=previous draw\n    Sigma    = exp(lnsd1)^2,   v10, v1t \\ v10, exp(lnsd0)^2, v0t \\ v1t, v0t, 1   //Constructing existing variance matrix\n    SigmaHat = exp(lnsd1Hat)^2,v10, v1t \\ v10, exp(lnsd0)^2, v0t \\ v1t, v0t, 1   //Proposed variance matrix with new lnsd1\n\n    if ( hasmissing(cholesky(SigmaHat)) == 0 ) {                                 //Reject if new variance mat is not posdef\n        val    = ln_L((ey1, ey0, et), Sigma)                                     //Compute data likelihood at old Sigma\n        valHat = ln_L((ey1, ey0, et), SigmaHat)                                  //Compute data likelihood at new Sigma\n        rat = valHat - val                                                       //Log ratio of two values.\n        alpha = min((exp(rat), 1))                                               //alpha is actual ratio\n        if (runiform(1,1,0,1) < alpha) lnsd1 = lnsd1Hat                          //accept draw with prob. alpha\n            prosd1 = exp(gam*(alpha - asta))*prosd1                              //adjust proposal dist. up or down \n    }\n    else {\n        prosd1 = exp(-asta*gam)*prosd1                                           //adjustment if rejected due to nonposdef\n    }')

get_ipython().run_cell_magic('stata', '--mata', '    prosd1 = 1\n    prosd0 = 1\n    prov10 = 1\n    prov1t = 1\n    prov0t = 1\n    draws = 10000')

get_ipython().run_cell_magic('stata', '--mata', "for (i=1;i<=10000;i++) {\n\n    Sig12 = (v10, v1t)\n    Sig22 = exp(lnsd0)^2, v0t \\ v0t, 1\n    Sig22m1 = invsym(Sig22)\n\n    CM = rowsum( (Sig12*Sig22m1):*(ey0, et) )\n    CV = exp(lnsd1)^2 - Sig12*Sig22m1*Sig12'\n\n    mc1   = m1 + CM\n    y1Hat = mc1 + rnormal(rows(y),1,0,1)*sqrt(CV)\n    y1    = tr:*y + (1 :-tr):*y1Hat\n    \n    mb1 = XX*X'(y1 - CM)\n    vb1 = CV*XX\n    b1 = mb1 + cholesky(vb1)*rnormal(cols(b1), 1, 0, 1)\n    b1 = b1'\n\n    m1 = X*b1'\n    ey1 = (y1 - m1)\n\n    Sig12 = (v10, v0t)\n    Sig22 = exp(lnsd1)^2, v1t \\ v1t, 1\n    Sig22m1 = invsym(Sig22)\n    \n    CM = rowsum( (Sig12*Sig22m1):*(ey1, et) )\n    CV = exp(lnsd0)^2 - Sig12*Sig22m1*Sig12'\n    \n    mc0   = m0 + CM\n    y0Hat = mc0 + rnormal(rows(y),1,0,1)*sqrt(CV)\n    y0 = (1 :- tr):*y + tr:*y0Hat\n        \n    mb0 = XX*X'(y0 - CM)\n    vb0 = CV*XX\n    b0 = mb0 + cholesky(vb0)*rnormal(cols(b0), 1, 0, 1)\n    b0 = b0'\n\n    m0 = X*b0'\n    ey0 = (y0 - m0)\t\n    \n    Sig12 = (v1t, v0t)\n    Sig22 = exp(lnsd1)^2, v10 \\ v10, exp(lnsd0)^2\n    Sig22m1 = invsym(Sig22)\n\n    CM = rowsum( (Sig12*Sig22m1):*(ey1, ey0) )\n    CV = 1 - Sig12*Sig22m1*Sig12'\n    \n    mct = mt + CM\n    et  = CV*invnormstab( normal(-mct/CV) + (1 :- normal(-mct/CV)):*runiform(rows(mct),1) )\n    ent = CV*invnormstab( normal(-mct/CV):*runiform(rows(mct),1) )\n    z = mct + et:*tr + ent:*(1 :- tr)\n\n    meane = WW*W'(z - CM)\n    vare  = CV*WW\n    e = meane + cholesky(vare)*rnormal(cols(e), 1, 0, 1)\n    e = e'\n    \n    mt = W*e'\n    et = (z - mt)\n\n    gam = 1/i^delta\n\n    lnsd1Hat = lnsd1 + rnormal(1,1,0,1)*prosd1\n    Sigma    = exp(lnsd1)^2,   v10, v1t \\ v10, exp(lnsd0)^2, v0t \\ v1t, v0t, 1\n    SigmaHat = exp(lnsd1Hat)^2,v10, v1t \\ v10, exp(lnsd0)^2, v0t \\ v1t, v0t, 1 \n\n    if ( hasmissing(cholesky(SigmaHat)) == 0 ) {\n        val    = ln_L((ey1, ey0, et), Sigma)\n        valHat = ln_L((ey1, ey0, et), SigmaHat)\n        rat = valHat - val\n        alpha = min((exp(rat), 1))\n        if (runiform(1,1,0,1) < alpha) lnsd1 = lnsd1Hat\n        prosd1 = exp(gam*(alpha - asta))*prosd1\n    }\n    else {\n        prosd1 = exp(-asta*gam)*prosd1\n    }\n    \n    lnsd0Hat = lnsd0 + rnormal(1,1,0,1)*prosd0\n    Sigma    = exp(lnsd1)^2, v10, v1t \\ v10, exp(lnsd0)^2,    v0t \\ v1t, v0t, 1\n    SigmaHat = exp(lnsd1)^2, v10, v1t \\ v10, exp(lnsd0Hat)^2, v0t \\ v1t, v0t, 1 \n\n    if ( hasmissing(cholesky(SigmaHat)) == 0 ) {\n        val    = ln_L((ey1, ey0, et), Sigma)\n        valHat = ln_L((ey1, ey0, et), SigmaHat)\n        rat = valHat - val\n        alpha = min((exp(rat), 1))\n        if (runiform(1,1,0,1) < alpha) lnsd0 = lnsd0Hat\n        prosd0 = exp(gam*(alpha - asta))*prosd0\n    }\n    else {\n        prosd0 = exp(-asta*gam)*prosd0\n    }\n\n    v10Hat = v10 + rnormal(1,1,0,1)*prov10\n    Sigma    = exp(lnsd1)^2, v10,    v1t \\ v10,    exp(lnsd0)^2, v0t \\ v1t, v0t, 1\n    SigmaHat = exp(lnsd1)^2, v10Hat, v1t \\ v10Hat, exp(lnsd0)^2, v0t \\ v1t, v0t, 1 \n\n    if ( hasmissing(cholesky(SigmaHat)) == 0 ) {\n        val    = ln_L((ey1, ey0, et), Sigma)\n        valHat = ln_L((ey1, ey0, et), SigmaHat)\n        rat = valHat - val\n        alpha = min((exp(rat), 1))\n        if (runiform(1,1,0,1) < alpha) v10 = v10Hat\n        prov10 = exp(gam*(alpha - asta))*prov10\n    }\n    else {\n        prov10 = exp(-asta*gam)*prov10\n    }\n\n\n    v1tHat = v1t + rnormal(1,1,0,1)*prov1t\n    Sigma    = exp(lnsd1)^2, v10, v1t    \\ v10, exp(lnsd0)^2, v0t \\ v1t,    v0t, 1\n    SigmaHat = exp(lnsd1)^2, v10, v1tHat \\ v10, exp(lnsd0)^2, v0t \\ v1tHat, v0t, 1 \n\n    if ( hasmissing(cholesky(SigmaHat)) == 0 ) {\n        val    = ln_L((ey1, ey0, et), Sigma)\n        valHat = ln_L((ey1, ey0, et), SigmaHat)\n        rat = valHat - val\n        alpha = min((exp(rat), 1))\n        if (runiform(1,1,0,1) < alpha) v1t = v1tHat\n        prov1t = exp(gam*(alpha - asta))*prov1t\n    }\n    else {\n        prov1t = exp(-asta*gam)*prov1t\n    }\n    \n    v0tHat = v0t + rnormal(1,1,0,1)*prov0t\n    Sigma    = exp(lnsd1)^2, v10, v1t \\ v10, exp(lnsd0)^2, v0t    \\ v1t, v0t,    1\n    SigmaHat = exp(lnsd1)^2, v10, v1t \\ v10, exp(lnsd0)^2, v0tHat \\ v1t, v0tHat, 1 \n\n    if ( hasmissing(cholesky(SigmaHat)) == 0 ) {\n        val    = ln_L((ey1, ey0, et), Sigma)\n        valHat = ln_L((ey1, ey0, et), SigmaHat)\n        rat = valHat - val\n        alpha = min((exp(rat), 1))\n        if (runiform(1,1,0,1) < alpha) v0t = v0tHat\n        prov0t = exp(gam*(alpha - asta))*prov0t\n    }\n    else {\n        prov0t = exp(-asta*gam)*prov0t\n    }\n\n    b1Hold = b1Hold \\ b1\n    b0Hold = b0Hold \\ b0\n    eHold = eHold \\ e\n    \n    sd1Hold = sd1Hold \\ lnsd1\n    sd0Hold = sd0Hold \\ lnsd0\n    v10Hold = v10Hold \\ v10\n    v1tHold = v1tHold \\ v1t\n    v0tHold = v0tHold \\ v0t\t\n\n}")

get_ipython().run_cell_magic('stata', '', '\npreserve\nclear\n\ngetmata (b1*) = b1Hold\ngetmata (b0*) = b0Hold\ngetmata (e*)= eHold\ngetmata (sd1*) = sd1Hold\ngetmata (sd0*) = sd0Hold \ngetmata (v10*) = v10Hold\ngetmata (v1z*) = v1tHold \ngetmata (v0z*) = v0tHold')

get_ipython().run_cell_magic('stata', '', 'sum b1*\nmat list binit1')

get_ipython().run_cell_magic('stata', '', 'sum b0*\nmat list binit0\n\nsum e*\nmat list einit')

get_ipython().run_cell_magic('stata', '--mata', '')


