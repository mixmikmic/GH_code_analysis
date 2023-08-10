Km, Kp, Ks = 0.001, 4, 0.5

Cxo=0.003  #cmol\L
Cso=6      #cmol/L

def ferm_eq(N,t):
    Cx, Cs, Cp, V = N[0]/N[3],N[1]/N[3],N[2]/N[3],N[3]  # defining concentration and volume, Note that N=[Nx,Ns,Np,V]
    r=response_fun([Cx, Cs, Cp])                        # same as before
    
    return []  # Complete this on your own 

V, Csf, Qf = 1000, 400/30, 8  #L, cmol/L, L/h
Cso, Cxo = 0.1, 0.003          #cmol/l

# from numpy import asarray
# Ci=(N[:, :3].T/N[:, 3]).T                   #first 3 elements of N devided by 4th element to get Ci, 
# r=numpy.asarray([res_fun(C) for C in Ci])



