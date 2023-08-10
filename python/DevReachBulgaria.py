subscription_secret = 'INSERT YOUR KEY HERE'

import http.client, urllib.request, urllib.parse, urllib.error, base64, json

# Replace with your subscription key.
subscription_key = subscription_secret

# Declare region.
uri_base = 'westcentralus.api.cognitive.microsoft.com'

headers = {
    # Request headers.
    'Content-Type': 'application/json',
    'Ocp-Apim-Subscription-Key': subscription_key,
}

params = urllib.parse.urlencode({
    # Request parameters. 
    'language': 'unk',
    'detectOrientation ': 'true',
})

# The URL of a JPEG image containing text.
body = "{'url':'http://4.bp.blogspot.com/-azhSS2fK6Ts/VhQaNJvmFrI/AAAAAAAAA4U/EAFtonT7sY8/s1600/SDC10865.JPG'}"

try:
    # Execute the REST API call and get the response.
    conn = http.client.HTTPSConnection('westcentralus.api.cognitive.microsoft.com')
    conn.request("POST", "/vision/v1.0/ocr?%s" % params, body, headers)
    response = conn.getresponse()
    data = response.read()

    # 'data' contains the JSON data. The following formats the JSON data for display.
    parsed = json.loads(data.decode())
    print ("Response:")
    print (json.dumps(parsed, sort_keys=True, indent=2))
    conn.close()

except Exception as e:
    print('Error:')
    print(e)

import requests

# Get the key from tab Keys on Azure portal
key = "INSERT YOUR KEY HERE" 

url4authentication = 'https://api.cognitive.microsoft.com/sts/v1.0/issueToken'
headers4authentication = {'Ocp-Apim-Subscription-Key': key}
resp4authentication = requests.post(url4authentication, headers=headers4authentication)
token = resp4authentication.text

# Call the Text Translate API
text = """

бяло вино,

вкусна пържола, 

пиле Алфредо

"""
come = "bg"
to = "en"

url4translate = 'https://api.microsofttranslator.com/v2/http.svc/Translate'
params = {'appid': 'Bearer '+token, 'text': text, 'from': come, 'to': to}
headers4translate = {'Accept': 'application/xml'}
resp4translate = requests.get(url4translate, params=params, headers=headers4translate)
print(resp4translate.text)



