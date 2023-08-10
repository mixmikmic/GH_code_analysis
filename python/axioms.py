from proveit.basiclogic.boolean.boolOps import Implies, Not, Or
from proveit.basiclogic import Equals
from proveit.basiclogic.boolean.quantifiers import Forall
from proveit.common import a, b, c, x, y, z, f, fx, fy, fa, fb, xEtc, yEtc, zEtc
from proveit.number.arithmeticOps import Add, Subtract, Multiply, GreaterThan, LessThan,     GreaterThanEquals, LessThanEquals, Summation, DiscreteContiguousSet
from proveit.number.numberSets import Integers
from proveit.number.common import one
import proveit.specialStatementMagic # for %begin_axioms and %end_axioms

get_ipython().magic('begin_axioms')

addAssoc = Forall([xEtc,yEtc,zEtc],
                  Equals(
                        Add(
                                xEtc,yEtc,zEtc),
                        Add(
                                xEtc,Add(yEtc),zEtc)
                        ),
                  )
addAssoc

multAssoc = Forall([xEtc,yEtc,zEtc],
                  Equals(
                        Multiply(
                                xEtc,yEtc,zEtc),
                        Multiply(
                                xEtc,Multiply(yEtc),zEtc)
                        )
                  )
multAssoc

reverseGreaterThanEquals = Forall((x, y), Implies(GreaterThanEquals(x, y), LessThanEquals(y, x)))
reverseGreaterThanEquals

reverseLessThanEquals = Forall((x, y), Implies(LessThanEquals(x, y), GreaterThanEquals(y, x)))
reverseLessThanEquals

reverseGreaterThan = Forall((x, y), Implies(GreaterThan(x, y), LessThan(y, x)))
reverseGreaterThan

reverseLessThan = Forall((x, y), Implies(LessThan(x, y), GreaterThan(y, x)))
reverseLessThan

greaterThanEqualsDef = Forall((x,y), Implies(GreaterThanEquals(x,y), Or(GreaterThan(x,y),Equals(x,y))))
greaterThanEqualsDef

lessThanEqualsDef = Forall((x,y), Implies(LessThanEquals(x,y), Or(LessThan(x,y),Equals(x,y))))
lessThanEqualsDef

lessThanTransLessThanRight = Forall((x,y,z),
                               Implies(LessThan(x,y),
                                      Implies(LessThan(y,z),
                                             LessThan(x,z))))
lessThanTransLessThanRight

lessThanTransLessThanEqualsRight = Forall((x,y,z),
                               Implies(LessThan(x,y),
                                      Implies(LessThanEquals(y,z),
                                             LessThan(x,z))))
lessThanTransLessThanEqualsRight

lessThanTransLessThanLeft = Forall((x,y,z),
                               Implies(LessThan(x,y),
                                      Implies(LessThan(z,x),
                                             LessThan(z,y))))
lessThanTransLessThanLeft

lessThanTransLessThanEqualsLeft = Forall((x,y,z),
                               Implies(LessThan(x,y),
                                      Implies(LessThanEquals(z,x),
                                             LessThan(z,y))))
lessThanTransLessThanEqualsLeft

lessThanEqualsTransLessThanRight = Forall((x,y,z),
                               Implies(LessThanEquals(x,y),
                                      Implies(LessThan(y,z),
                                             LessThan(x,z))))
lessThanEqualsTransLessThanRight

lessThanEqualsTransLessThanEqualsRight = Forall((x,y,z),
                               Implies(LessThanEquals(x,y),
                                      Implies(LessThanEquals(y,z),
                                             LessThanEquals(x,z))))
lessThanEqualsTransLessThanEqualsRight

lessThanEqualsTransLessThanLeft = Forall((x,y,z),
                               Implies(LessThanEquals(x,y),
                                      Implies(LessThan(z,x),
                                             LessThan(z,y))))
lessThanEqualsTransLessThanLeft

lessThanEqualsTransLessThanEqualsLeft = Forall((x,y,z),
                               Implies(LessThanEquals(x,y),
                                      Implies(LessThanEquals(z,x),
                                             LessThanEquals(z,y))))
lessThanEqualsTransLessThanEqualsLeft

greaterThanTransGreaterThanRight = Forall((x,y,z),
                                    Implies(GreaterThan(x,y),
                                           Implies(GreaterThan(y,z),
                                                  GreaterThan(x,z))))
greaterThanTransGreaterThanRight

greaterThanTransGreaterThanEqualsRight = Forall((x,y,z),
                                    Implies(GreaterThan(x,y),
                                           Implies(GreaterThanEquals(y,z),
                                                  GreaterThan(x,z))))
greaterThanTransGreaterThanEqualsRight

greaterThanTransGreaterThanLeft = Forall((x,y,z),
                                    Implies(GreaterThan(x,y),
                                           Implies(GreaterThan(z,x),
                                                  GreaterThan(z,y))))
greaterThanTransGreaterThanLeft

greaterThanTransGreaterThanEqualsLeft = Forall((x,y,z),
                                    Implies(GreaterThan(x,y),
                                           Implies(GreaterThanEquals(z,x),
                                                  GreaterThan(z,y))))
greaterThanTransGreaterThanEqualsLeft

greaterThanEqualsTransGreaterThanRight = Forall((x,y,z),
                                               Implies(GreaterThanEquals(x,y),
                                                      Implies(GreaterThan(y,z),
                                                             GreaterThan(x,z))))
greaterThanEqualsTransGreaterThanRight

greaterThanEqualsTransGreaterThanEqualsRight = Forall((x,y,z),
                                               Implies(GreaterThanEquals(x,y),
                                                      Implies(GreaterThanEquals(y,z),
                                                             GreaterThanEquals(x,z))))
greaterThanEqualsTransGreaterThanEqualsRight

greaterThanEqualsTransGreaterThanLeft = Forall((x,y,z),
                                               Implies(GreaterThanEquals(x,y),
                                                      Implies(GreaterThan(z,x),
                                                             GreaterThan(z,y))))
greaterThanEqualsTransGreaterThanLeft

greaterThanEqualsTransGreaterThanEqualsLeft = Forall((x,y,z),
                                               Implies(GreaterThanEquals(x,y),
                                                      Implies(GreaterThanEquals(z,x),
                                                             GreaterThanEquals(z,y))))
greaterThanEqualsTransGreaterThanEqualsLeft

sumSingle = Forall(f, Forall(a,
                              Equals(Summation(x,fx,DiscreteContiguousSet(a,a)),
                                     fa),
                              domain=Integers))
sumSingle

sumSplitLast = Forall(f, 
                      Forall([a,b],
                             Equals(Summation(x,fx,DiscreteContiguousSet(a,b)),
                                    Add(Summation(x,fx,DiscreteContiguousSet(a,Subtract(b, one))),
                                       fb)),
                             domain=Integers, conditions=[LessThan(a, b)]))
sumSplitLast

get_ipython().magic('end_axioms')



