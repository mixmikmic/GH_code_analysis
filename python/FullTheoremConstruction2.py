from qpeScratch import *

def wrap1(arg1):
    return Forall([m,n,t,eps],
             arg1,
             [In(m,Zp),In(n,Zp),In(t,Zp),In(eps,zeroToOne),Equals(t,tFunc_n_eps)])

def wrap2(arg2):
    return Forall([U,u,phi],
       arg2,[In(U,SUm),
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

#Full theorem is below



