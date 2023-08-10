import requests
import json, os

def spaceprint(val, cnt):
    leng = len(str(val))
    cnt = cnt - leng
    restr = ""
    for i in range(cnt):
        restr += " "
    restr = restr+str(val)
    return restr

url = "{0}:{1}".format(os.environ['HOSTNAME'] , "8000")
nn_id = "mro001"
net_type = "cnn"
wf_ver_id = 1
wf_ver_id = str(wf_ver_id)

# CNN Network WorkFlow Node : Network Config Setup
resp = requests.put('http://' + url + '/api/v1/type/wf/state/netconf/detail/cnn/nnid/'+nn_id+'/ver/'+wf_ver_id+'/node/netconf_node/',
                     json={
                         "param":{"traincnt": 1
                                  ,"epoch": 1
                                  ,"batch_size":1000000
                                  ,"predictcnt": 5
                                  ,"predlog": "N" # T:Ture, F:False, A:True&False, N:None
                         },
                         "config": {"num_classes":10,
                                    "learnrate": 0.001,
                                    "layeroutputs":32,
                                    "net_type":"cnn",
                                    "eval_type":"category",
                                    "optimizer":"AdamOptimizer" #RMSPropOptimizer
                                     }
                         ,"layer1": {"active": "relu",
                                     "cnnfilter": [3, 3],
                                     "cnnstride": [1, 1],
                                     "maxpoolmatrix": [2, 2],
                                     "maxpoolstride": [2, 2],
                                     "padding": "SAME",
                                     "droprate": "0.8",
                                     "layercnt":4
                                    }
                         ,"layer2": {"active": "relu",
                                     "cnnfilter": [3, 3],
                                     "cnnstride": [1, 1],
                                     "maxpoolmatrix": [1, 1],
                                     "maxpoolstride": [2, 2],
                                     "padding": "SAME",
                                     "droprate": "0.8",
                                     "layercnt":1
                                    }
                         ,"out": {"active": "softmax",
                                   "node_out": 625,
                                   "padding": "SAME"
                                }
                         ,"labels":[]
                        })

netconf = json.loads(resp.json())
print("insert workflow node conf info netconf result : {0}".format(netconf))

# CNN Network WorkFlow Node :  Eval Config Setup
node = "eval_node"
resp = requests.put('http://' + url + '/api/v1/type/wf/state/netconf/detail/cnn/nnid/'+nn_id+'/ver/'+wf_ver_id+'/node/'+node+'/'
                    ,json={})
evalconf = json.loads(resp.json())

# yolo min image size 385 and %7 = 0
datajson = {"preprocess": {"x_size": 32,
                           "y_size": 32,
                           "channel":3,
                           "filesize": 1000000,
                           "yolo": "n"}
            }

# CNN Network WorkFlow Node :  Data Config Setup
# yolo min image size 385 and %7 = 0
node = "datasrc"
resp = requests.put('http://' + url + '/api/v1/type/wf/state/imgdata/src/local/form/file/prg/source/nnid/'+nn_id+'/ver/'+wf_ver_id+'/node/'+node+'/',
                     json=datajson)
dataconf = json.loads(resp.json())
print("")
print("insert workflow node conf info dataconf result : {0}".format(dataconf))

# CNN Network WorkFlow Node :  Eval Data Config Setup
node = 'evaldata'
resp = requests.put('http://' + url + '/api/v1/type/wf/state/imgdata/src/local/form/file/prg/source/nnid/'+nn_id+'/ver/'+wf_ver_id+'/node/'+node+'/'
                    ,json=datajson)
edataconf = json.loads(resp.json())

resp = requests.post('http://' + url + '/api/v1/type/runmanager/state/train/nnid/'+nn_id+'/ver/'+wf_ver_id+'/')
data = json.loads(resp.json())

if data == None:
    print(data)
else:
    try:
        if data["status"] == "404":
            print(data["result"])
    except:
        for train in data:
            if train != None and train != "" and train != {} and train != "status" and train != "result":
                try:
                    for tr in train["TrainResult"]:
                        print(tr)
                except:
                    maxcnt = 0
                    line = ""
                    for label in train["labels"]:
                        if maxcnt<len(label)+2:
                            maxcnt = len(label)+2

                    for i in range(len(train["labels"])):
                        for j in range(maxcnt+4):
                            line += "="

                    label_sub = []
                    for label in train["labels"]:
                        label = spaceprint(label,maxcnt)
                        label_sub.append(label)

                    space = ""
                    for s in range(maxcnt):
                        space +=" "

                    print(space, label_sub)
                    print(space, line)
                    for i in range(len(train["labels"])):
                        truecnt = 0
                        totcnt = 0
                        predict_sub = []
                        for j in range(len(train["predicts"][i])):
                            pred = spaceprint(train["predicts"][i][j],maxcnt)

                            predict_sub.append(pred)
                            totcnt += int(pred)
                            # print(train["labels"].index(train["labels"][i]))
                            if train["labels"].index(train["labels"][i]) == j:
                                truecnt = int(pred)
                        if totcnt == 0:
                            percent = 0
                        else:
                            percent = round(truecnt/totcnt*100,2)
                        print(spaceprint(train["labels"][i],maxcnt), predict_sub, str(percent)+"%")

files = {
         'files000001':  open('/home/dev/hoyai/demo/data/airplane/1air.jpg','rb')
        ,'files000002':  open('/home/dev/hoyai/demo/data/airplane/2air.jpg','rb')
        ,'files000003':  open('/home/dev/hoyai/demo/data/bolt/1bolt.jpg','rb')
        ,'files000004':  open('/home/dev/hoyai/demo/data/bolt/2bolt.jpg','rb')
        ,'files000005':  open('/home/dev/hoyai/demo/data/car/1car.jpg','rb')
        ,'files000006':  open('/home/dev/hoyai/demo/data/car/2car.jpg','rb')
        ,'files000007':  open('/home/dev/hoyai/demo/data/glove/1glove.jpg','rb')
        ,'files000008':  open('/home/dev/hoyai/demo/data/glove/2glove.jpg','rb')
        ,'files000009':  open('/home/dev/hoyai/demo/data/motor/1motor.jpg','rb')
        ,'files000010':  open('/home/dev/hoyai/demo/data/motor/2motor.jpg','rb')
        }
restURL = 'http://' + url + '/api/v1/type/service/state/predict/type/cnn/nnid/'+nn_id+'/ver/'+wf_ver_id+'/'

resp = requests.post(restURL,
                     files=files
                     )
data = json.loads(resp.json())

for train in data:
    print("FileName = "+train)
    print(data[train]['key'])
    print(data[train]['val'])
    print('')



