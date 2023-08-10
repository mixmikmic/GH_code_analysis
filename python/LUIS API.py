import requests
import json

config = json.load(open("./config.json"))

luis_url = f"https://westus.api.cognitive.microsoft.com/luis/v2.0/apps/{config['LuisAppID']}"

headers = {
    "Ocp-Apim-Subscription-Key": config['SubscriptionKey']
}

params = {
    "q": "Create note"
}

response = requests.get(luis_url, headers=headers, params=params)
response.json()

params = {
    "q": "Add to shopping list"
}

response = requests.get(luis_url, headers=headers, params=params)
response.json()

