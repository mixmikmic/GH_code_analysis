import requests
import re

# Call API to access credit card data
url = 'https://www.moneyhero.com.hk/api/credit-card/v2/cards/all?lang=en&pageSize=1000'

response = requests.get(url)
data = response.json()

# Parse the json reponse and get the name of the credit card and url of the credit card image
cards = data['featuredCards'] + data['cards']
card_names =  [card['name'] for card in cards]
card_images = [card['image'] for card in cards]
card_images_name = [re.search('([^/]+$)', image).group(1) for image in card_images]

# Download raw image
import urllib
for i in range(len(card_images)):
    urllib.request.urlretrieve(card_images[i], 'input/train_raw_card_image/'+card_images_name[i])

df = pd.DataFrame({ 'name': pd.Series(card_names),
                    'image_name': pd.Series(card_images_name)})

df.sort_values(by = 'image_name').to_csv('input/credit-card.csv')

# Reduce the size of the image

# Reading the image (Skip if already resized)
card_images_raw = list(map(lambda img: misc.imread('input/train_raw_card_image/'+img)[:,:,:3], df2.image_name))
# Resize the image so that they have the same dimension
card_images_resize = list(map(lambda img: misc.imresize(img, (56, 92, 3)), card_images_raw))
# Saver the image if necessary
for i in range(len(card_images_resize)):
    misc.imsave('images/'+df.image_name[i], card_images_resize[i])

