import json , gzip
def read_gziped_json(file_address):
    with gzip.GzipFile(file_address, 'r') as f:
        json_bytes = f.read()
    json_str = json_bytes.decode('utf-8')
    data = json.loads(json_str)
    return data

multiclass_corpus = read_gziped_json("data/project_dc_multiclass.gz")
print (type(multiclass_corpus))
print (multiclass_corpus.keys())
print ("number of training set examples:" , len (multiclass_corpus["training"]))
print ("number of test set     examples:" , len (multiclass_corpus["test"]))

for example in multiclass_corpus["training"][0:3]:
    print (example , "\n")

for example in multiclass_corpus["test"][0:3]:
    print (example , "\n")

multilabel_corpus = read_gziped_json("data/project_dc_multilabel.gz")
print ("number of examples:" , len (multilabel_corpus))
for example in multilabel_corpus[0:3]:
    print (example , "\n")

x = [example for example in multilabel_corpus if len(example[2])==1]
for example in x[0:3]:
    print (example, "\n")

x = [example for example in multilabel_corpus if len(example[2])>5]
for example in x[0:3]:
    print (example, "\n")



