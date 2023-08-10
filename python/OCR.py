API_KEY = 'AIzaSyD-S4VpFHwMQcvweVrx03g6YywP5iYtGLA'

import cv2
from base64 import b64encode  
from os import makedirs  
from os.path import join, basename  
from sys import argv  
import json  
import requests
from matplotlib import pyplot as plt

ENDPOINT_URL = 'https://vision.googleapis.com/v1/images:annotate'  
RESULTS_DIR = 'jsons'

def make_image_data_list(image_filenames):  
    """
    image_filenames is a list of filename strings
    Returns a list of dicts formatted as the Vision API
        needs them to be
    """
    img_requests = []
    for imgname in image_filenames:
        with open(imgname, 'rb') as f:
            ctxt = b64encode(f.read()).decode()
            img_requests.append({
                    'image': {'content': ctxt},
                    'features': [{
                        'type': 'TEXT_DETECTION',
                        'maxResults': 1
                    }]
            })
    return img_requests

def make_image_data(image_filenames):  
    """Returns the image data lists as bytes"""
    imgdict = make_image_data_list(image_filenames)
    return json.dumps({"requests": imgdict }).encode()


def request_ocr(api_key, image_filenames):  
    response = requests.post(ENDPOINT_URL,
                            data=make_image_data(image_filenames),
                            params={'key': api_key},
                            headers={'Content-Type': 'application/json'})
    return response

image_filenames = ['label.jpg']
response = request_ocr(API_KEY, image_filenames)

response.json()

responseJson = response.json()

for response in responseJson['responses'][0][u'textAnnotations']:
    print response['boundingPoly']

if response.status_code != 200 or response.json().get('error'):
    print(response.text)
else:
    for idx, resp in enumerate(response.json()['responses']):
        # save to JSON file
        imgname = image_filenames[idx]
        jpath = join(RESULTS_DIR, basename(imgname) + '.json')
        with open(jpath, 'w') as f:
            datatxt = json.dumps(resp, indent=2)
            print("Wrote", len(datatxt), "bytes to", jpath)
            f.write(datatxt)

        # print the plaintext to screen for convenience
        print("---------------------------------------------")
        t = resp['textAnnotations'][0]
        print("    Bounding Polygon:")
        print(t['boundingPoly'])
        print("    Text:")
        print(t['description'])

image_filenames = ['label2.jpg']
response = request_ocr(API_KEY, image_filenames)

if response.status_code != 200 or response.json().get('error'):
    print(response.text)
else:
    for idx, resp in enumerate(response.json()['responses']):
        # save to JSON file
        imgname = image_filenames[idx]
        jpath = join(RESULTS_DIR, basename(imgname) + '.json')
        with open(jpath, 'w') as f:
            datatxt = json.dumps(resp, indent=2)
            print("Wrote", len(datatxt), "bytes to", jpath)
            f.write(datatxt)

        # print the plaintext to screen for convenience
        print("---------------------------------------------")
        t = resp['textAnnotations'][0]
        print("    Bounding Polygon:")
        print(t['boundingPoly'])
        print("    Text:")
        print(t['description'])

print(t['description'])

description = t['description'].splitlines()
print description[0]
print description[1]

price = description[1].split()
print price[0][1:]
print price[1]

img = cv2.imread('label2.jpg')

cv2.rectangle(img, ( t['boundingPoly']['vertices'][0]['x'], t['boundingPoly']['vertices'][0]['y']), ( t['boundingPoly']['vertices'][2]['x'], t['boundingPoly']['vertices'][2]['y']), (0,255,0),1)

plt.imshow(img)
plt.show()

img = cv2.imread('label2.jpg')
for t in responseJson['responses'][0][u'textAnnotations']:
    cv2.rectangle(img, ( t['boundingPoly']['vertices'][0]['x'], t['boundingPoly']['vertices'][0]['y']), ( t['boundingPoly']['vertices'][2]['x'], t['boundingPoly']['vertices'][2]['y']), (0,255,0),1)

plt.imshow(img)
plt.show()

from skimage.feature import hog
from skimage import data, color, exposure, io

image = color.rgb2gray(io.imread('label2.jpg'))

fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualise=True)

plt.imshow(image, cmap=plt.cm.gray)
plt.show()

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

plt.imshow(hog_image_rescaled, cmap=plt.cm.gray)
plt.show()

img = cv2.imread('label3.jpg')
image_filenames = ['label3.jpg']
API_KEY = 'AIzaSyCzemskXK5JabNNLy6fnwMD-hQfws0jFE4'

response = request_ocr(API_KEY, image_filenames)

if response.status_code != 200 or response.json().get('error'):
    print(response.text)
else:
    for idx, resp in enumerate(response.json()['responses']):
        # save to JSON file
        imgname = image_filenames[idx]
        jpath = join(RESULTS_DIR, basename(imgname) + '.json')
        with open(jpath, 'w') as f:
            datatxt = json.dumps(resp, indent=2)
            print("Wrote", len(datatxt), "bytes to", jpath)
            f.write(datatxt)

        # print the plaintext to screen for convenience
        print("---------------------------------------------")
        t = resp['textAnnotations'][0]
        print("    Bounding Polygon:")
        print(t['boundingPoly'])
        print("    Text:")
        print(t['description'])

img = cv2.imread('label3.jpg')
for t in responseJson['responses'][0][u'textAnnotations']:
    cv2.rectangle(img, ( t['boundingPoly']['vertices'][0]['x'], t['boundingPoly']['vertices'][0]['y']), ( t['boundingPoly']['vertices'][2]['x'], t['boundingPoly']['vertices'][2]['y']), (0,255,0),1)



image_filenames = ['label.jpg']

response = request_ocr(API_KEY, image_filenames)

