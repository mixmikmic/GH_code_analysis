import pandas as pd
import boto3

comprehend = boto3.client(service_name='comprehend')

text = "It is raining today in Seattle"
print('Calling DetectSentiment')
print(json.dumps(comprehend.detect_sentiment(Text=text, LanguageCode='en'), sort_keys=True, indent=4))
print('End of DetectSentiment\n')

path = "/Users/noahgift/Desktop/review_polarity/txt_sentoken/neg/cv000_29416.txt"
doc1 = open(path, "r")
output = doc1.readlines()

output[2]

print(json.dumps(comprehend.detect_sentiment(Text=output[2], LanguageCode='en'), sort_keys=True, indent=4))

whole_doc = ', '.join(map(str, output))

print(json.dumps(comprehend.detect_sentiment(Text=whole_doc, LanguageCode='en'), sort_keys=True, indent=4))



