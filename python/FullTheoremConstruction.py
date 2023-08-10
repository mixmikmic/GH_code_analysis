#import sys
#from proveit.expression import Literal, LATEX, STRING, Operation
##from proveit.statement import *
#from proveit.basiclogic.genericOps import AssociativeOperation, BinaryOperation, OperationOverInstances
#from proveit.everythingLiteral import EVERYTHING
from proveit.number.arithmeticOps import *
from proveit.number.variables import a,b,c,m,n,t,eps,phi,U,SUm,C2m,H,Hm,u,e,i,pi,k,l,zero,one,two,infinity
from proveit.number.variables import minusOne, minusTwo,Z,Zp,R,zeroToOne,tFunc,tFunc_n_eps,QPE,QPEfunc
#import proveit.number.variables as var
from proveit.basiclogic.boolean.quantifiers import Forall
from proveit.basiclogic.set.setOps import In
from proveit.basiclogic import Equals

def wrap1(arg1):
    return Forall([m,n,t,eps],
             arg1,
             conditions=[In(m,Zp),In(n,Zp),In(t,Zp),In(eps,zeroToOne),Equals(t,tFunc_n_eps)])

wrap1(Z)

def wrap2(arg2):
    return Forall([U,u,phi],
       arg2,conditions=[In(U,SUm),
          In(u,Hm),
          In(phi,zeroToOne),
            Equals(
                Multiply(U,u),
                Multiply(
                    Exponentiate(
                        e,
                        Multiply(two,pi,i,phi)
                        ),
                    u)
                )
             ]
            )

PrArg = LessThanEquals(
    Abs(
        Subtract(
            QPEfunc,
            phi)
    ),
    Exponentiate(
        two,
    Multiply(minusOne,
             n)
        ))

print Pr(PrArg)

print wrap1(wrap2( GreaterThanEquals(Pr(PrArg),Subtract(one,eps)))).formatted(LATEX)



