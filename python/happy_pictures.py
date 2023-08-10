import requests
import json
import facebook  # pip install git+https://github.com/pythonforfacebook/facebook-sdk.git
import time
import os
import urllib

from IPython.display import Image
from collections import defaultdict

from api_keys import emotion_exford_api_key, facebook_api_key

EMOTION_API_URL = 'https://api.projectoxford.ai/emotion/v1.0/recognize'
EMOTION_HEADERS = {
    'Ocp-Apim-Subscription-Key': emotion_exford_api_key
}

class OxfordAPIException(Exception):
    pass


def evaluate_image(url):
    """
    Returns dictionary of different emotions and face rectangles
    """    
    data_encoded = json.dumps({'url': url})
    r = requests.post(EMOTION_API_URL, data=data_encoded, headers=EMOTION_HEADERS)
    data = r.json()
    
    if not r.status_code == requests.codes.ok:
        if data.get('statusCode', None) == 429:
            time.sleep(60)  # there is some one minute rate limit
            return evaluate_image(url)
        raise OxfordAPIException(data)
    return r.json()
    

graph_api = facebook.GraphAPI(facebook_api_key)
photos_backup_file = 'photos_fb_backup.json'

def scrap_objects(object_path, after=None):
    """
    Downloads FB material from given object_path following the paginators
    """
    response = graph_api.get_object(object_path, after=after)
    data = response['data']
    after = response.get('paging', {}).get('cursors', {}).get('after', None)
    
    if after:
        time.sleep(1)  # very primitive way to slow down API
        data += scrap_objects(object_path, after=response['paging']['cursors']['after'])
    return data

if os.path.exists(photos_backup_file):
    with open(photos_backup_file, 'r') as bck_file:
        photos = json.loads(bck_file.read())
else:
    photos = scrap_objects('/me/photos')
    with open(photos_backup_file, 'w') as bck_file:
        bck_file.write(json.dumps(photos))

def max_resolution(images):
    """Finds the best quality photo in facebook dictionary of different sizes"""
    return max(images, key=lambda image: image['width'])

def photo_url(photo):
    return max_resolution(photo['images'])['source']

def photo_file_path(photo):
    basename = os.path.basename(photo_url(photo))
    return os.path.join('photos', basename[:24])

def download_photo(photo):
    url_to_grab = photo_url(photo)
    basename = os.path.basename(url_to_grab)
    urllib.request.urlretrieve(url_to_grab, os.path.join('photos', basename[:24]))

try:
    for photo in photos: download_photo(photo)
except:
    pass

processed_backup_file = 'photos_with_emotions.json'

def process_photo(photo):
    url = max_resolution(photo['images'])['source']
    photo['emotions'] = evaluate_image(url)
    
if os.path.exists(processed_backup_file):
    with open(processed_backup_file, 'r') as bck_file:
        photos = json.loads(bck_file.read())
else:    
    for photo in photos: process_photo(photo)
    with open(processed_backup_file, 'w') as bck_file:
        bck_file.write(json.dumps(photos))

def show_photo(photo):
    file_path = photo_file_path(photo)
    if os.path.exists(file_path):
        return Image(filename=file_path, format='png')    
    return Image(url=max_resolution(photo['images'])['source'])

def sums_persons_emotions(photo, exclude=None):
    """
    Sums all the emotions of all the people in the photo 
    """
    final_score = defaultdict(int)  # int() always returns 0
    
    for person in photo['emotions']:
        for emotion, score in person['scores'].items():
            if exclude and emotion in exclude:
                continue
            final_score[emotion] += score
    return final_score

def sum_of_all_emotions(photo):
    return sum(sums_persons_emotions(photo, exclude=['neutral']).values())


most_emotional_photo = max(photos, key=sum_of_all_emotions)
show_photo(most_emotional_photo)

sums_persons_emotions(most_emotional_photo)

def average_emotions(photo, exclude=None):
    emotion_sum = sums_persons_emotions(photo, exclude=exclude)
    number_of_people = len(photo['emotions'])
    if not number_of_people:
        return {}
    
    return {
        emotion: score / number_of_people
        for emotion, score in emotion_sum.items()
    }  

def emotions_per_person(photo):
    return sum(average_emotions(photo, exclude='neutral').values())

most_emotions_per_person_photo = max(photos, key=emotions_per_person) 
show_photo(most_emotions_per_person_photo)

most_emotions_per_person_photo['emotions']

def sum_of_single_emotion(photo, emotion):
    return sums_persons_emotions(photo)[emotion]

def average_of_emotion(photo, emotion):
    if not len(photo['emotions']):
        return 0
    
    return sum_of_single_emotion(photo, emotion) / len(photo['emotions'])

def find_max_photo_emotion_sum(emotion):
    return max(photos, key=lambda photo: sum_of_single_emotion(photo, emotion))

def find_max_photo_emotion_average(emotion):
    return max(photos, key=lambda photo: average_of_emotion(photo, emotion))


def sorted_by_emotion_average(emotion):
    return sorted(photos, reverse=True, key=lambda photo: average_of_emotion(photo, emotion))

show_photo(find_max_photo_emotion_average('sadness'))

disusting_photo = find_max_photo_emotion_average('disgust')
show_photo(disusting_photo)

disusting_photo['emotions']

show_photo(sorted_by_emotion_average('disgust')[2])

show_photo(find_max_photo_emotion_average('surprise'))

show_photo(find_max_photo_emotion_average('neutral'))

show_photo(sorted_by_emotion_average('neutral')[1])

contempting = find_max_photo_emotion_average('contempt')
show_photo(contempting)

contempting['emotions']

show_photo(sorted_by_emotion_average('contempt')[1])

show_photo(find_max_photo_emotion_average('fear'))

show_photo(find_max_photo_emotion_average('anger'))

show_photo(sorted_by_emotion_average('anger')[1])

all_emotions = defaultdict(int)
for average in map(average_emotions, photos):
    for emotion, score in average.items():
        all_emotions[emotion] += score
all_emotions

