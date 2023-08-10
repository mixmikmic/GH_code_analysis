from proveit.common import fx
# fx is f(x) which is simply an Operation with a Variable operator and a Variable operand,
# but that is irrelevant.  It was chosen arbitrarily.
fx.exprInfo()

fxTruth = fx.prove(assumptions=[fx])
fxTruth

from proveit import ProofFailure
try:
    fx.prove()
except ProofFailure as e:
    print "EXPECTED ERROR: ", e

fxTruth.expr

fxTruth.assumptions

print dir(fxTruth)

fxTruth.operands

print fxTruth.__class__ # this is a giveaway that it is not an actual Expression

print fxTruth.expr.__class__ # here it is

fxTruth.proof()

fxTruth.proof().provenTruth

fxTruth.proof().requiredProofs # no requirements for {f(x)} |- f(x)

from proveit import defaults
defaults.assumptions = [fx]
newFxTruth = fx.prove()
newFxTruth

newFxTruth._proof

# But if you want to turn off this storage, you can.
from proveit import storage
from proveit.common import gy
storage.directory = None
gy # not retrieved or placed into storage (this take extra time generate the image)

# Or you can use an alternative storage directory
import os
try:
    os.mkdir('test_storage')
except:
    pass
storage.directory = 'test_storage'
from proveit.common import Px
Px

# You can see that it did store some things in the '.pv_it' folder of this new directory.
# These are stored by hash values with collision protection (collision are astronomically
# unlikely at these hash string lengths, but it never hurts to be on the safe side).
os.listdir(os.path.join('test_storage', '.pv_it')) 

# You can also clear the storage if you wish:
storage.clear() # or equivalent just delete the '.pv_it' folder.

try:
    os.listdir(os.path.join('test_storage', '.pv_it')) 
except OSError as e:
    print "EXPECTED ERROR:", e

# it is still stored in memory as the png attribute
if Px.png is not None:
    print "png is stored"
Px

os.rmdir('test_storage') # let's delete this test storage directory now



