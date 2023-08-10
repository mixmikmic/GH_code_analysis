from googleapiclient import discovery
import httplib2
from oauth2client.client import GoogleCredentials
import base64
from PIL import Image
from PIL import ImageDraw

# Set up a connection to the Google service
DISCOVERY_URL='https://{api}.googleapis.com/$discovery/rest?version={apiVersion}'
credentials = GoogleCredentials.get_application_default()
service = discovery.build('vision', 'v1', credentials=credentials, discoveryServiceUrl=DISCOVERY_URL)

# load the input images
input_filename = "dalai-lama.jpg"
image = open(input_filename,'rb')
image_content = image.read()

# fire off request for face detection from Google
batch_request = [{
    'image': {
        'content': base64.b64encode(image_content)
        },
    'features': [{
        'type': 'FACE_DETECTION',
        'maxResults': 4,
        }]
    }]
request = service.images().annotate(body={
    'requests': batch_request,
    })
response = request.execute()
print('Found %s face%s' % (len(response['responses']), '' if len(response['responses']) == 1 else 's'))

#print response
# print out emotions for each one
for result in response['responses']:
    for annotation in result['faceAnnotations']:
        for emotion in ['joy','sorrow','surprise','anger']:
            print "%s: %s" % (emotion, annotation[emotion+'Likelihood'])
        



