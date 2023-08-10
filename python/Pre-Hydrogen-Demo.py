import warnings
import numpy as np
import phe as paillier
from sonar.contracts import ModelRepository,Model
from syft.he.Paillier import KeyPair
from syft.nn.linear import LinearClassifier
import numpy as np
from sklearn.datasets import load_diabetes

def get_balance(account):
    return repo.web3.fromWei(repo.web3.eth.getBalance(account),'ether')

warnings.filterwarnings('ignore')

# for the purpose of the simulation, we're going to split our dataset up amongst
# the relevant simulated users

diabetes = load_diabetes()
y = diabetes.target
X = diabetes.data

validation = (X[0:42],y[0:42])
anonymous_diabetes_users = (X[42:],y[42:])

# we're also going to initialize the model trainer smart contract, which in the
# real world would already be on the blockchain (managing other contracts) before
# the simulation begins

# ATTENTION: copy paste the correct address (NOT THE DEFAULT SEEN HERE) from truffle migrate output.
repo = ModelRepository('0xf30068fb49616db7d5afb89862d6b40d11389327', ipfs_host='localhost', web3_host='localhost') # blockchain hosted model repository



# we're going to set aside 400 accounts for our 400 patients
# Let's go ahead and pair each data point with each patient's 
# address so that we know we don't get them confused
patient_addresses = repo.web3.eth.accounts[1:40]
anonymous_diabetics = list(zip(patient_addresses,
                               anonymous_diabetes_users[0],
                               anonymous_diabetes_users[1]))

# we're going to set aside 1 account for Cure Diabetes Inc
cure_diabetes_inc = repo.web3.eth.accounts[0]

pubkey,prikey = KeyPair().generate(n_length=1024)
diabetes_classifier = LinearClassifier(desc="DiabetesClassifier",n_inputs=10,n_labels=1)
initial_error = diabetes_classifier.evaluate(validation[0],validation[1])
diabetes_classifier.encrypt(pubkey)

diabetes_model = Model(owner=cure_diabetes_inc,
                       syft_obj = diabetes_classifier,
                       bounty = 1,
                       initial_error = initial_error,
                       target_error = 10000
                      )

model_id = repo.submit_model(diabetes_model)

cure_diabetes_inc

model_id

model = repo[model_id]

diabetic_address,input_data,target_data = anonymous_diabetics[0]

repo[model_id].submit_gradient(diabetic_address,input_data,target_data)

repo[model_id]

old_balance = get_balance(diabetic_address)
print(old_balance)

new_error = repo[model_id].evaluate_gradient(cure_diabetes_inc,repo[model_id][0],prikey,pubkey,validation[0],validation[1])

new_error

new_balance = get_balance(diabetic_address)
incentive = new_balance - old_balance
print(incentive)

model

for i,(addr, input, target) in enumerate(anonymous_diabetics):
    try:
        
        model = repo[model_id]
        
        # patient is doing this
        model.submit_gradient(addr,input,target)
        
        # Cure Diabetes Inc does this
        old_balance = get_balance(addr)
        new_error = model.evaluate_gradient(cure_diabetes_inc,model[i+1],prikey,pubkey,validation[0],validation[1],alpha=2)
        print("new error = "+str(new_error))
        incentive = round(get_balance(addr) - old_balance,5)
        print("incentive = "+str(incentive))
    except:
        "Connection Reset"

