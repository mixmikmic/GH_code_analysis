import time
import giphy_client
from giphy_client.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = giphy_client.DefaultApi()
api_key = 'dc6zaTOxFJmzC' # str | Giphy API Key.
q = 'microscope' # str | Search query term or prhase.
limit = 25 # int | The maximum number of records to return. (optional) (default to 25)
offset = 0 # int | An optional results offset. Defaults to 0. (optional) (default to 0)
rating = 'g' # str | Filters results by specified rating. (optional)
lang = 'en' # str | Specify default country for regional content; use a 2-letter ISO 639-1 country code. See list of supported languages <a href = \"../language-support\">here</a>. (optional)
fmt = 'json' # str | Used to indicate the expected response format. Default is Json. (optional) (default to json)


data = []

for i in range(30):
    try: 
        # Search Endpoint
        api_response = api_instance.gifs_search_get(api_key, q, limit=limit, offset=min((i*limit,100)), rating=rating, lang=lang, fmt=fmt)
        data += api_response.data
    except ApiException as e:
        print("Exception when calling DefaultApi->gifs_search_get: %s\n" % e)

urls = []
for tmp in data:
    url = tmp.images.downsized.url
    print(url)
    urls.append(url)

urls

import pandas as pd

ps = pd.DataFrame(data=urls, columns=["giffile"])

ps.to_csv("giphy_urls.csv")

get_ipython().run_line_magic('pinfo', 'pd.Series')





