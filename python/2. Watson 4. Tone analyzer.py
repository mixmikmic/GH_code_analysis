# The code was removed by DSX for sharing.

import json
from watson_developer_cloud import ToneAnalyzerV3


tone_analyzer = ToneAnalyzerV3(
    username=credentials_1['username'],
    password=credentials_1['password'],
    version='2016-02-11')

print(json.dumps(tone_analyzer.tone(text='I am very happy'), indent=2))

