import requests
import operator
import os
import io
import base64
import pathlib
from PIL import Image
import json
from pprint import PrettyPrinter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
get_ipython().run_line_magic('matplotlib', 'inline')
import sys
print(sys.version)

#HEADERS = {'content-type': 'application/json'} ; # charset=utf-8
IP = "localhost"
PORT = "8080"
URL = "http://{}:{}/facebox/check".format(IP, PORT)
print(URL)
IMG_FILE = "thebeatles.jpg"
FIG_SIZE = (12, 8)
ATTR_ENTITY_ID = 'entity_id'
VALID_ENTITY_ID = 'image_processing.facebox_demo_camera'

def print_json(json_data):
    PrettyPrinter().pprint(json_data)
    
def encode_image(image):
        """base64 encode an image stream."""
        base64_img = base64.b64encode(image).decode('ascii')
        return {"base64": base64_img}

MOCK_FACES = {'confidence': 0.5812028911604818,
              'id': 'john.jpg',
              'matched': True,
              'name': 'John Lennon',
              'rect': {'height': 75, 'left': 63, 'top': 262, 'width': 74}
              }

MOCK_FACES

MOCK_FACES[ATTR_ENTITY_ID] = VALID_ENTITY_ID

MOCK_FACES

base64.b64encode(b'test').decode('ascii')

img = plt.imread(IMG_FILE)
fig, ax = plt.subplots(figsize=FIG_SIZE)
ax.imshow(img);

get_ipython().run_cell_magic('time', '', 'IMG_FILE = "/Users/robincole/.homeassistant/images/thebeatles.jpg"\nfile = {\n    \'file\': (IMG_FILE, open(IMG_FILE, \'rb\')),\n}\n\nresponse = requests.post(URL, files=file).json()')

print_json(response)

response['facesCount']

image = Image.open(IMG_FILE, mode='r') # Create a JpegImageFile object

with io.BytesIO() as output:
    with Image.open(IMG_FILE) as img:
        img.save(output, 'BMP')
    data = output.getvalue()

# data is a bytes object

response = requests.post(URL, json=encode_image(data), timeout=9).json()

print_json(response['success'])

def get_matched_faces(response):
    """Return the name and confidence of matched faces."""
    return {face['name']: round(face['confidence'], 3) 
            for face in response['faces'] 
            if face['matched']}

get_matched_faces(response)

def get_bounding_boxes(response):
    """Return the bounding boxes of faces."""
    bounding_boxes = []
    for face in response['faces']:
        bounding_boxes.append(face['rect'])
    return bounding_boxes

bounding_boxes = get_bounding_boxes(response)
bounding_boxes

img = plt.imread(IMG_FILE)
fig, ax = plt.subplots(figsize=FIG_SIZE)
ax.imshow(img);

for box in bounding_boxes:
    x = box['left']
    y = box['top']
    width = box['width']
    height = box['height']

    rect = patches.Rectangle((x,y), width, height, linewidth=5, edgecolor='r', facecolor='none')
    ax.add_patch(rect);

plt.savefig('facebox.png')

def save_boxes_image(img_file, bounding_boxes):
    """Take an image file and dict of bounding boxes and save the boxes on a copy of the image."""
    img = plt.imread(img_file)
    fig, ax = plt.subplots()
    ax.imshow(img);
    for box in bounding_boxes:
        x = box['left']
        y = box['top']
        width = box['width']
        height = box['height']

        rect = patches.Rectangle((x,y), width, height, linewidth=5, edgecolor='r', facecolor='none')
        ax.add_patch(rect);

    plt.savefig("boxed_image.png")

save_boxes_image(IMG_FILE, bounding_boxes)

