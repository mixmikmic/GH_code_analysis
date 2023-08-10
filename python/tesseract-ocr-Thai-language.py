from PIL import Image
import pytesseract

img_path = 'data-text-img/BangkokThai_(vectorized).svg.png'
Image.open(img_path)

# Python - using pytesseract for Thai language

txtImg = Image.open(img_path)
text = pytesseract.image_to_string(txtImg, 'tha')

print text

img_path = 'data-text-img/thai-text.png'
Image.open(img_path)

txtImg = Image.open(img_path)
text = pytesseract.image_to_string(txtImg, 'tha')

print text

img_path = 'data-text-img/thai-text-004.jpg'
Image.open(img_path)

txtImg = Image.open(img_path)
text = pytesseract.image_to_string(txtImg, 'tha')

print text

img_path = 'data-text-img/thai-text-005.jpg'
Image.open(img_path)

txtImg = Image.open(img_path)
text = pytesseract.image_to_string(txtImg, 'tha')

print text

