import pandas as pd

transactions = pd.read_csv("transactions.csv", index_col=0, names=["block_id", "is_coinbase"])
transactions.index.name = "tr_id"

issues = [52534,   # 2 coinbases! have to drop it
          11181,   # tries to mint more 10 satoshis that are not in fees...
          12042,   # Double spend attack for output 7998, spent at input 521...
          15567, # Tries to spend an output that does not exist (yet) 
          30223,   # Tries to double spend the same output (21928) in the same TX
          56565,  # Tries to double spend output 65403
          72902, # Outputs higher (10 satoshis) then input
          75047, # Tries to have a negative 50 BTC output...
          79885,  # Tries to spend an output from a block with 2 coinbases...
          88755,  # Tries to spend the output that it creates to create it
          96607]  # Tries to spend an output that has not been created yet.
    
print("Will remove {} blocks".format(len(issues)))

problematic_blocks = transactions.block_id.isin(issues)
print("Removing {} transactions".format(sum(problematic_blocks)))
tx_to_remove = transactions[transactions.block_id.isin(issues)].index.values
print(tx_to_remove)
transactions =  transactions[~problematic_blocks] # 2 coinbases for 52354, we will have to drop it
# Now ce can simply deal with transactions! 
# Let's also re


# Let's add info on outputs and inputs to transactions
inputs = pd.read_csv("inputs.csv", index_col=0, names=["tr_id", "output_id"])
outputs = pd.read_csv("outputs.csv", index_col=0, names=["tr_id", "pk_id" ,"value"])

# Let's also drop those that refer to to the 52354 here.
inputs = inputs[~inputs.tr_id.isin(tx_to_remove)]
outputs = outputs[~outputs.tr_id.isin(tx_to_remove)]


# Check for two coinbases !
two_coinbase = sum(transactions[transactions.is_coinbase == 1].block_id.value_counts() > 1)
assert(not two_coinbase)

# Let's add columns for inputs and outputs in the transaction table

input_summary = {tr_id: [] for tr_id in transactions.index}
output_summary =  {tr_id: [] for tr_id in transactions.index}

for id, tr_id in inputs.tr_id.items():
    input_summary[tr_id].append(id)
for id, tr_id in outputs.tr_id.items():
    output_summary[tr_id].append(id)
    
transactions["inputs"] = pd.Series(input_summary)
transactions["outputs"] = pd.Series(output_summary)

def handle_coinbase(transaction):
    # Perform basic checks
    assert(not transaction.inputs)
    try:
        assert(len(transaction.outputs) > 0)
    except:
        print(transaction)
        print(1/0)

    coinbase = outputs.loc[transaction.outputs]
    if coinbase.shape[0] > 1:
        try:
            assert(coinbase.value.sum() == coinbase_value)
        except:
            to_check.append(transaction.block_id)
        for output, pk, value in zip(coinbase.index, coinbase.pk_id, coinbase.value):
            utxo[output] = [pk, value]
        return
    
    try:
        assert(int(coinbase["value"]) == coinbase_value)
    except:
        to_check.append(transaction.block_id)
    try:
        pk_id = int(coinbase.pk_id)
    except:
        print(transaction)
        print(coinbase)
        print(1/0)
    output_id = coinbase.index[0]
    utxo[output_id] = [pk_id, int(coinbase["value"])]

        
def handle_transaction(transaction):
    
    assert(transaction.inputs)
    assert(transaction.outputs)
    
    all_outputs = outputs.loc[transaction.outputs]
    value_output = all_outputs.value.sum()
    
    all_inputs = inputs.loc[transaction.inputs].output_id
    try:
        value_input = sum(utxo[i][1] for i in all_inputs)
    except:
        print(inputs.loc[transaction.inputs])
        print(transaction)
        print(1/0)
    
    try:
        assert(value_input == value_output)
    except:
        diff = value_input - value_output
        if not transaction.block_id in to_check:
            print(transaction)
            print(diff)
            print(1/0)
        if diff < 0:
            print(transaction)
            print(1/0)
    
    # We can now do the transaction!
    for i in all_inputs:
        try:
            utxo.pop(i)
        except:
            print(all_inputs)
            print(transaction)
            print(1/0)
    
    for output_id, pk_id, value in zip(all_outputs.index, all_outputs.pk_id, all_outputs.value):
        utxo[output_id] = [pk_id, value]

utxo = {} # mapping UTXO to [owner, value]
coinbase_value = 50*10**8
to_check = [] # we check fees externally as its a block level feature

for i, coinbase in enumerate(transactions.is_coinbase):
    if (i+1) % 10**4 == 0:
        print("Handled {} transactions".format(i))
    if coinbase:
        handle_coinbase(transactions.iloc[i])
    else:
        handle_transaction(transactions.iloc[i])

len(utxo)

val = [i[1] for i in utxo.values()]
max(val)/10**8

pk_ids = pd.Series(range(outputs.pk_id.nunique()), index=outputs.pk_id.unique())

for _, (input_, output_) in transactions[["inputs", "outputs"]].iterrows():
    if not (_ + 1) % 10000:
        print(_ + 1)
        print("Number distinct values: ", len(set(pk_ids)))
    
    if len(input_) > 1 or (len(output_) == 1 and len(input_)):
        pk_input = list(outputs.loc[inputs.loc[input_].output_id].pk_id)
        pk_output = list(outputs.loc[output_].pk_id)
    else:
        continue

    if len(input_) > 1:
        for pk in pk_input[1:]:
            clus_1, clus_2 = pk_ids.loc[pk_input[0]], pk_ids.loc[pk]
            if clus_1 != clus_2:
                pk_ids = pk_ids.replace(clus_1, clus_2) 
                
    if len(output_) == 1 and len(input_):
        for i in range(len(pk_input)):
            clus_1, clus_2 = pk_ids.loc[pk_input[0]], pk_ids.loc[pk_output[0]]
            if clus_1 != clus_2: 
                pk_ids = pk_ids.replace(clus_1, clus_2)

possessions_check = {}
for _, (pk, val) in outputs[~outputs.index.isin(inputs.output_id)][["pk_id", "value"]].iterrows():
    uf_pkid = pk_ids.loc[pk]
    possessions_check[uf_pkid] = possessions_check.get(uf_pkid, 0) + val

max(possessions_check.values())/10**8

entity = max(possessions_check, key=possessions_check.get)
entity

controlled = []
for key, val in pk_ids.items():
    if val == entity:
        controlled.append(key)

min(controlled)

senders = []
for j, tx in enumerate(outputs[outputs.pk_id.isin(controlled)].tr_id):
    if not (j+1)%10**4:
        print(j+1)
    origin = inputs.loc[transactions.loc[tx].inputs].output_id
    tx_series = outputs.loc[origin]
    if (~tx_series.pk_id.isin(controlled)).all():
        senders.append([tx, tx_series.value.sum()])

senders = pd.Series([s[1] for s in senders], index=[s[0] for s in senders])

print(senders.max()/10**8)
print(senders.argmax())

transactions.loc[98122]

