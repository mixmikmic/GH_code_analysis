import warnings
import numpy as np
import phe as paillier
from sonar.contracts import ModelRepository,Model
from syft.he.paillier.keys import KeyPair
from syft.nn.linear import LinearClassifier
import numpy as np
from sklearn.datasets import load_diabetes

def get_balance(account):
    return repo.web3.fromWei(repo.web3.eth.getBalance(account),'ether')

warnings.filterwarnings('ignore')

from sonar import GridClient

# for the purpose of the simulation, we're going to split our dataset up amongst
# the relevant simulated users

diabetes = load_diabetes()
y = diabetes.target
X = diabetes.data

validation = (X[0:5],y[0:5])

# we're also going to initialize the model trainer smart contract, which in the
# real world would already be on the blockchain (managing other contracts) before
# the simulation begins

# ATTENTION: copy paste the correct address (NOT THE DEFAULT SEEN HERE) from truffle migrate output.
grid = GridClient('0xcc37157ad0f60637025e33d68eeeaf546a360dc6', ipfs_host='localhost', web3_host='localhost') # blockchain hosted model repository

diabetes_classifier = LinearClassifier(desc="DiabetesClassifier",n_inputs=10,n_labels=1)

diabetes_model = Model(owner=cure_diabetes_inc,
                       syft_obj = diabetes_classifier,
                       bounty = 1000,
                       training_input = X,
                       training_target = y
                      )
model_id = grid.submit_model(diabetes_model)

grid.model_status(model_id)

grid.model_status(model_id)

grid.model_status(model_id)

trained_diabetes_model = grid.get_model(model_id)





