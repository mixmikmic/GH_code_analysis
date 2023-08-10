# Load libraries
import numpy as np
import xgboost as xgb
from flask import Flask, abort, jsonify, request
import pandas as pd
import json

bst = xgb.Booster() #init model
bst.load_model("../0002.model") # load data

df = pd.read_csv("../datasample/base1run1.csv")
df.head()

m1=df.as_matrix()[0:3,1:]
if len(m1.shape) == 1:
    inputdata=xgb.DMatrix(m1[np.newaxis,:],feature_names=bst.feature_names)
else:
    inputdata=xgb.DMatrix(m1,feature_names=bst.feature_names)

preds = bst.predict(inputdata)
#print(preds)
l1=preds[:,0].tolist()
l2=df.as_matrix()[0:3,0]
pd.DataFrame({'Time':l2,'predictions':l1}).to_json(orient='records')

app = Flask(__name__)

@app.route('/api',methods=['POST'])
def make_predict():
    dataobject = request.get_json(force=True)
    #print(dataobject)
    predict_request = list(dataobject['data'].values())
    predict_request = np.array(predict_request[1:-1], dtype='f')
    if len(predict_request.shape) == 1:
        
        times = np.array(predict_request[0], dtype='f')
        inputdata=xgb.DMatrix(predict_request[np.newaxis,:],feature_names=bst.feature_names)
    else:
        times = np.array(predict_request[:,0], dtype='f')
        inputdata=xgb.DMatrix(predict_request,feature_names=bst.feature_names)

    preds = bst.predict(inputdata)
    #print(preds)
    output = preds[:,0].tolist()
    
    outputdf = pd.DataFrame({'Time':times,'predictions':output}).to_json(orient='records')
    
    returnmsg = jsonify(deviceId = dataobject['deviceId'],
                        datetime = dataobject['datetime'],
                        protocol = dataobject['protocol'],
                        batchID = dataobject['data']['batchID'],
                        results=json.loads(outputdf))
    
    return returnmsg

@app.route("/")
def hello():
    return "Python Flask POST endpoint."

#
# Running app in Jupyter notebook:
#

from werkzeug.serving import run_simple
run_simple('localhost', 9000, app)



