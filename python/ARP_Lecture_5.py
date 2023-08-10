D = 1.2019E+00, 2.9205E-01
SigmaA = 1.0596E-02, 1.2871E-01
nuSigmaF = 7.9522E-03, 1.9312E-01
SigmaS21 = 2.0769E-02
phi = 3.1349E+00,  4.9894E-01

k_balance = (phi[0]*nuSigmaF[0]+phi[1]*nuSigmaF[1])/(phi[0]*SigmaA[0]+phi[1]*SigmaA[1])
k_balance

k_oo = 1/(SigmaS21+SigmaA[0]) * (nuSigmaF[0]+ nuSigmaF[1]*SigmaS21/SigmaA[1])
k_oo



