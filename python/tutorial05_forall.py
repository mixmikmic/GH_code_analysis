# Let us create a basic Forall expression and examine it
from proveit import Forall
from proveit.common import x, P, Px, Q, Qx, R, Rx, S
basicForallExpr = Forall(x, Px, conditions=[Qx, Rx], domain=S)
basicForallExpr

basicForallExpr.exprInfo()

basicForallExpr.instanceVars # list of Variables

basicForallExpr.instanceExpr

basicForallExpr.conditions

basicForallExpr.domain

from proveit import Operation, InSet, ExpressionList
from proveit.common import fy
assumptions = ExpressionList(basicForallExpr, InSet(fy, S), Operation(Q, fy), Operation(R, fy))
assumptions

basicForallSpec = basicForallExpr.specialize({x:fy}, assumptions=assumptions)
basicForallSpec

basicForallSpec.proof()

from proveit import ProofFailure
try:
    basicForallExpr.specialize({x:fy}, assumptions=assumptions[1:])
except ProofFailure as e:
    print "EXPECTED ERROR:", e

from proveit import SpecializationFailure
try:
    basicForallExpr.specialize({x:fy}, assumptions=assumptions[:1]+assumptions[2:])
except SpecializationFailure as e:
    print "EXPECTED ERROR:", e

try:
    basicForallExpr.specialize({x:fy}, assumptions=assumptions[:2]+assumptions[3:])
except SpecializationFailure as e:
    print "EXPECTED ERROR:", e

try:
    basicForallExpr.specialize({x:fy}, assumptions=assumptions[:3])
except SpecializationFailure as e:
    print "EXPECTED ERROR:", e

basicForallExpr

try:
    basicForallExpr.specialize({x:fy, Q:R}, assumptions=assumptions)
except SpecializationFailure as e:
    print "EXPECTED ERROR:", e

noDomainForallExpr = Forall(x, Px, conditions=[Qx])
noDomainForallExpr

print noDomainForallExpr.domain

noDomainForallExpr.exprInfo()

noDomainForallExpr.specialize({x:fy}, assumptions=[noDomainForallExpr, Operation(Q, fy)])

from proveit.common import Pxy, y, fy
from proveit.logic import Exists, NotEquals
forallExistsExpr = Forall(x, Exists(y, NotEquals(x, y)))
forallExistsExpr

Exists(y, Pxy).exprInfo()

from proveit import ScopingViolation
try:
    forallExistsExpr.specialize({x:y}, assumptions={forallExistsExpr})
except ScopingViolation as e:
    print "EXPECTED ERROR:", e

from proveit import ScopingViolation
try:
    forallExistsExpr.specialize({x:fy}, assumptions={forallExistsExpr})
except ScopingViolation as e:
    print "EXPECTED ERROR:", e

from proveit.logic import And
redundantInstanceVarExpr = Forall(x, And(Px, Forall(x, Qx)))
redundantInstanceVarExpr

# specializing the outer x does not and should not change the inner x which is treated as a distinct Variable
redundantInstanceVarExpr.specialize({x:fy}, assumptions={redundantInstanceVarExpr})

from proveit.logic.equality._axioms_ import substitution
substitution

substitution.proof()

x_eq_y = substitution.conditions[0]
x_eq_y

from proveit.common import f, g
operatorSubstitution = substitution.specialize({f:g}, assumptions=[x_eq_y])
operatorSubstitution

operatorSubstitution.proof()

from proveit.logic import Equals
from proveit.common import a, b
a_eq_b = Equals(a, b)
operandSubstitution = substitution.specialize({x:a, y:b}, assumptions=[a_eq_b])
operandSubstitution

operandSubstitution.proof()

from proveit import Lambda
from proveit.number import Add
operationSubstitution = substitution.specialize({f:Lambda(x, Add(x, a))}, assumptions=[x_eq_y])
operationSubstitution

operationSubstitution.proof()

from proveit.common import fx
operationSubstitution2 = substitution.specialize({fx:Add(x, a)}, assumptions=[x_eq_y])
operationSubstitution2

operationSubstitution2.proof()

from proveit.number import LessThan
from proveit.common import z, Pxyz
nestedForall = Forall(x, Forall(y, Forall(z, Pxyz, conditions=[LessThan(z, Add(x, y))])))
nestedForall

nestedForallSpec1 = nestedForall.specialize(assumptions=[nestedForall])
nestedForallSpec1

nestedForallSpec2 = nestedForallSpec1.specialize()
nestedForallSpec2

nestedForallSpec3 = nestedForallSpec2.specialize(assumptions=[nestedForallSpec2.conditions[0]])
nestedForallSpec3

nestedForallSpec3.proof()

assumptions = ExpressionList(nestedForall, nestedForallSpec2.conditions[0])
nestedForallSimultaneousSpec = nestedForall.specialize({z:z}, assumptions=assumptions)
nestedForallSimultaneousSpec

nestedForallSimultaneousSpec.proof()

nestedForallSpecAndRelab = nestedForall.specialize(specializeMap={y:y}, relabelMap={z:a}, assumptions=[nestedForall])
nestedForallSpecAndRelab

nestedForallSpecAndRelab.proof()

from proveit import RelabelingFailure
try:
    nestedForall.specialize({y:y}, {y:a}, assumptions=assumptions)
except RelabelingFailure as e:
    print "EXPECTED ERROR:", e

from proveit import RelabelingFailure
try:
    nestedForall.specialize({y:y}, {P:R}, assumptions=assumptions)
except RelabelingFailure as e:
    print "EXPECTED ERROR:", e

multiVarForall = Forall((x, y), Pxy, domain=S)
multiVarForall

assumptions = [multiVarForall, InSet(x, S), InSet(y, S)]
multiVarForallSpec = multiVarForall.specialize(assumptions=assumptions)
multiVarForallSpec

multiVarForallSpec.proof()

try:
    Forall((x, x), Px)
except ValueError as e:
    print 'EXPECTED ERROR:', e

from proveit.logic.boolean.disjunction._theorems_ import notOrIfNotAny
notOrIfNotAny

notOrIfNotAny.exprInfo()

from proveit.common import Amulti, c
from proveit.logic import Not
notOrIfNotAnySpec = notOrIfNotAny.specialize({Amulti:[a, b, c]}, assumptions=[Not(a), Not(b), Not(c)])
notOrIfNotAnySpec

notOrIfNotAnySpec.proof()

notOrIfNotAnySpec.exprInfo()

notOrIfNotAny.specialize({Amulti:[]})

from proveit.logic.boolean.disjunction._axioms_ import emptyDisjunction
emptyDisjunction

from proveit import Etcetera
from proveit.logic import inBool
from proveit.common import Bmulti
assumptions = [Not(a), Etcetera(Not(Bmulti)), inBool(a), Etcetera(inBool(Bmulti))]
notOrIfNotAnySpec2 = notOrIfNotAny.specialize({Amulti:(a, Bmulti)}, assumptions=assumptions)
notOrIfNotAnySpec2

notOrIfNotAnySpec

from proveit import GeneralizationFailure
try:
    notOrIfNotAnySpec.generalize((a, b, c))
except GeneralizationFailure as e:
    print 'EXPECTED ERROR:', e

notOrIfNotAnySpec.generalize((a, b, c), conditions=[Not(a), Not(b), Not(c)])

notOrIfNotAnySpec.generalize((a, b, c), conditions=[Not(a), Not(b), Not(c)], domain=S)

notOrIfNotAnySpec.generalize((a, b, c), conditions=[Not(a), Not(b), Not(c), Qx], domain=S)

notOrIfNotAnyNested = notOrIfNotAnySpec.generalize([[a], [b], [c]], domains=(P, R, S), conditions=[Not(a), Not(b), Not(c)])
notOrIfNotAnyNested

notOrIfNotAnyNested.proof()

notOrIfNotAnySpec.generalize([[a], [b], [c]], domains=(P, R, S), conditions=[Not(a), Not(c), Not(b)])

try:
    notOrIfNotAnySpec.generalize([[a], [b], [c]], domains=(P, R), conditions=[Not(a), Not(b), Not(c)])
except ValueError as e:
    print 'EXPECTED ERROR:', e

notOrIfNotAnySpec.generalize(a, conditions=[Not(a)])

try:
    notOrIfNotAnySpec.generalize(Qx, conditions=[Not(a)])
except ValueError as e:
    print 'EXPECTED ERROR:', e

try:
    notOrIfNotAnySpec.generalize([a, Qx], conditions=[Not(a)])
except ValueError as e:
    print 'EXPECTED ERROR:', e

notOrIfNotAnySpec2

from proveit.logic import Booleans
notOrIfNotAnySpec2.generalize((a, Bmulti), conditions=[Not(a), Etcetera(Not(Bmulti))], domain=Booleans)



