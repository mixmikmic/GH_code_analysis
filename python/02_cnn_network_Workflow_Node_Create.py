import requests
import json, os

url = "{0}:{1}".format(os.environ['HOSTNAME'] , "8000")

nn_id = "nn00011"
biz_cate = "ERP"
biz_sub_cate = "MRO"
nn_title = "MRO Image Classification"
nn_desc = "MRO Image Classification"
nn_wf_ver_info = "MRO Image Classification"
use_flag = "Y"
dirstr = "purpose"
config = "N"
network_type = "renet" #cnn, renet
node_sub_menu = "data_image"

# Create Network
resp = requests.post('http://' + url + '/api/v1/type/common/target/nninfo/nnid/'+nn_id+ '/',
                     json={
                         "nn_id": nn_id,
                         "biz_cate": biz_cate,
                         "biz_sub_cate": biz_sub_cate,
                         "nn_title" : nn_title,
                         "nn_desc": nn_desc,
                         "use_flag" : use_flag,
                         "dir": dirstr,
                         "config": config
                     })
data = json.loads(resp.json())
print("Create Network : {0}".format(data))

# Create Workflow 
resp = requests.post('http://' + url + '/api/v1/type/common/target/nninfo/nnid/'+nn_id+'/version/',
                     json={
                         "nn_def_list_info_nn_id": "",
                         "nn_wf_ver_info": nn_wf_ver_info,
                         "condition": "1",
                         "active_flag": "N"
                     })
data = json.loads(resp.json())

# Get Workflow Version
resp = requests.get('http://' + url + '/api/v1/type/common/target/nninfo/nnid/'+nn_id+'/version/')
data = json.loads(resp.json())

# get Active workflow version
wf_ver_id = 0
max_wf_ver_id = 0
data = sorted(data, key=lambda k: k['fields']['nn_wf_ver_id'])
for config in data:
    if config["fields"]["active_flag"] == "Y":
        wf_ver_id = config['fields']['nn_wf_ver_id']
print("Active Version Workflow ID=" + str(wf_ver_id))

# get Max workflow version
for config in data:
    if config["fields"]["nn_wf_ver_id"] > wf_ver_id:
        wf_ver_id = config["fields"]["nn_wf_ver_id"]

wf_ver_id = str(wf_ver_id)
print("Max Version Workflow ID=" + str(wf_ver_id))

# update workflow version info
resp = requests.put('http://' + url + '/api/v1/type/common/target/nninfo/nnid/'+nn_id+'/version/',
                     json={
                         "nn_wf_ver_id": wf_ver_id,
                         "nn_def_list_info_nn_id": "",
                         "nn_wf_ver_info": nn_wf_ver_info,
                         "condition": "1",
                         "active_flag": "Y"
                     })
data = json.loads(resp.json())
print("Update active workflow version info evaluation result : {0}".format(data))

# Create Workflow Node
resp = requests.post('http://' + url + '/api/v1/type/wf/target/init/mode/simple/'+nn_id+'/wfver/'+wf_ver_id+'/',
                     json={
                         "type": network_type
                     })
data = json.loads(resp.json())
print("Create Workflow Version Node : {0}".format(data))



