from miscpy.utils.sympyhelpers import *
init_printing()

th,ph,psi,thd,phd,psid = symbols('theta,phi,psi,thetadot,phidot,psidot')
w1,w2,w3 = symbols('omega_1,omega_2,omega_3')

cCi = Matrix(([cos(th),sin(th),0],[-sin(th),cos(th),0],[0,0,1]))
sCc = Matrix(([cos(ph),0,-sin(ph)],[0,1,0],[sin(ph), 0, cos(ph)]));cCi,sCc

sCi = sCc*cCi; sCi

aCi = Matrix(([cos(psi),sin(psi),0],[-sin(psi),cos(psi),0],[0,0,1]))
cCa = Matrix(([cos(th),0,-sin(th)],[0,1,0],[sin(th), 0, cos(th)]))
bCc = Matrix(([cos(ph),sin(ph),0],[-sin(ph),cos(ph),0],[0,0,1])); bCc, cCa, aCi

bCi = bCc*cCa*aCi; bCi

w = psid*bCi*Matrix([0,0,1]) + thd*bCc*Matrix([0,1,0]) +  phd*Matrix([0,0,1]); w

solve(w - Matrix([w1,w2,w3]),([thd,psid,phd]))

