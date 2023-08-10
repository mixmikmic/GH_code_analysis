service_name='REPLACE_WITH_SERVICE_NAME'

credentials=REPLACE_WITH_CREDENTIALS

vs={'streaming-analytics': [{'name': service_name, 'credentials': credentials}]}

import json
with open('vcap_services.json', 'w') as outfile:
    json.dump(vs, outfile)

