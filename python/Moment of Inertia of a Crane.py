from miscpy.utils.sympyhelpers import *
init_printing()

M,h,m1,m2,th1,th2,b,l1,l2 = symbols('M,h,m_1,m_2,theta_1,theta_2,beta,l_1,l_2')

I_O_cab = M*2/3*h**2/4*eye(3); I_O_cab

I_G1_m1_B = m1*l1**2/12*(eye(3) - diag(1,0,0)); I_G1_m1_B

aCb = Matrix(([cos(th1),-sin(th1),0],[sin(th1),cos(th1),0],[0,0,1])); aCb

I_G1_m1_A = aCb*I_G1_m1_B*aCb.transpose(); I_G1_m1_A

r_O_G1 = aCb*Matrix([-l1/2,0,0]) + Matrix([-h/2,0,0]); r_O_G1

I_O_m1_A = simplify(I_G1_m1_A + m1*((r_O_G1.transpose()*r_O_G1)[0]*eye(3) - r_O_G1*r_O_G1.transpose())); I_O_m1_A

I_G2_m2_C = m2*l2**2/12*(eye(3) - diag(1,0,0)); I_G2_m2_C

bCc = Matrix(([cos(b),sin(b),0],[-sin(b),cos(b),0],[0,0,1])); bCc

aCc = simplify(aCb*bCc); aCc

aCc = aCc.subs(b-th1,th2); aCc

I_G2_m2_A = aCc*I_G2_m2_C*aCc.transpose(); I_G2_m2_A

r_O_G2 = aCc*Matrix([-l2/2,0,0]) + aCb*Matrix([-l1,0,0]) + Matrix([-h/2,0,0]); r_O_G2

I_O_m2_A = simplify(I_G2_m2_A + m2*((r_O_G2.transpose()*r_O_G2)[0]*eye(3) - r_O_G2*r_O_G2.transpose())); I_O_m2_A

I_O_A = simplify(I_O_cab + I_O_m1_A + I_O_m2_A); I_O_A



