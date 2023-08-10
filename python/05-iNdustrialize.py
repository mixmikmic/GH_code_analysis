import os
os.chdir("..")

import json, requests

import pandas as pd
import numpy as np

from src.scrub import get_clean_iris
from src.deploy import generate_random_iris
from src.deploy import get_prediction

url = "http://127.0.0.1:5000/api"

payload = json.dumps({
    'sl': 3,
    'sw': 5,
    'pl': 2.75,
    'pw': 2.1,
    'algo': 'dt'
})

r = requests.post(url, payload)

json.loads(r.text)

df = generate_random_iris()

test_case = df.sample(1).iloc[0]
print(test_case)

get_prediction(test_case, 'dt', url=url)

df['predicted_iris_type'] = (df
 .apply(lambda r: get_prediction(r, 'lr', url=url).get('iris_type'), 
        axis=1)
)

df.head()



