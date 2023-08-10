from PIL import Image
import pytesseract

img_path = 'data-text-img/text-img.png'
Image.open(img_path)

txtImg = Image.open(img_path)
text = pytesseract.image_to_string(txtImg)

print text

img_path = 'data-text-img/text-img-01.png'
Image.open(img_path)

txtImg = Image.open(img_path)
text = pytesseract.image_to_string(txtImg)

print text

