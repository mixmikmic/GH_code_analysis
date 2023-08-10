from miscpy.utils.sympyhelpers import *
init_printing()
th,n1,n2,n3 = symbols('theta,n_1,n_2,n_3')

th = symbols('theta')

n = Matrix([0,0,1]);n

aCb = cos(th)*eye(3) + (1 - cos(th))*(n*n.T) + sin(th)*skew(n); aCb

aCb.T*n

n2 = Matrix([1,2,3])
n2 = n2/n2.norm()
n2

aCb2 = cos(th)*(eye(3) - n2*n2.T)+ n2*n2.T + sin(th)*skew(n2); aCb2

tmp = simplify(aCb2.T*n2); tmp

simplify(aCb2*tmp)



n1,n2,n3 = symbols('n_1,n_2,n_3')

ng = Matrix([n1,n2,n3]);ng

aCbg = simplify(cos(th)*eye(3) + (1 - cos(th))*(ng*ng.T) + sin(th)*skew(ng)); aCbg



