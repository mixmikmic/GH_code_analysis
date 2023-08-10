import os
import shutil
import requests
from os.path import join
from urllib3.exceptions import ReadTimeoutError
from requests.exceptions import RequestException

# credit to https://github.com/GalAvineri/ISIC-Archive-Downloader
def download_dataset(num_images, images_dir):
    
    url = 'https://isic-archive.com/api/v1/image?limit={0}&offset=0&sort=name&sortdir=1'.format(num_images)
    response = requests.get(url, stream=True)
    meta_data = response.json()
    ids = [meta_data[index]['_id'] for index in range(len(meta_data))]

    base_url_prefix = 'https://isic-archive.com/api/v1/image/'
    base_url_suffix = '/download?contentDisposition=inline'

    for id in ids:
        # Build the image url
        url_image = base_url_prefix + id + base_url_suffix
        # Build the description url
        url_desc = base_url_prefix + id

        # Download the image and description using the url
        # Sometimes their site isn't responding well, and than an error occurs,
        # So we will retry 10 seconds later and repeat until it succeeds
        succeeded = False
        while not succeeded:
            try:
                # Download the image and description
                response_image = requests.get(url_image, stream=True, timeout=20)
                response_desc = requests.get(url_desc, stream=True, timeout=20)
                # Validate the download status is ok
                response_image.raise_for_status()
                response_desc.raise_for_status()

                # Parse the description and write it into a file
                parsed_desc = response_desc.json()
                
                # Write the image into a file
                diagnosis = parsed_desc['meta']['clinical']['benign_malignant']
                img_path = join(images_dir, diagnosis)
                if not os.path.exists(img_path):
                    os.makedirs(img_path)
                img_file = join(img_path, '{0}.jpg'.format(parsed_desc['name']))    
                with open(img_file, 'wb') as imageFile:
                    shutil.copyfileobj(response_image.raw, imageFile)
                succeeded = True
            except RequestException as e:
                print(e)
            except ReadTimeoutError as e:
                print(e)
            except IOError as e:
                print(e)

## Specify the number of images to download
number_of_img_to_download = 10
## Specify the download directory
img_dir = '../../sample_data/isic_data'
download_dataset(number_of_img_to_download, img_dir)

