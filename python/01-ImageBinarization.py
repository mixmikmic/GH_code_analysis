get_ipython().magic('matplotlib inline')
import cv2
import matplotlib.pyplot as plt

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
def image_bin(image_gs):
    ret,image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
    return image_bin
def image_bin_otsu(image_gs):
    ret,image_bin = cv2.threshold(image_gs, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return image_bin
def image_bin_adaptive(image_gs):
    image_bin = cv2.adaptiveThreshold(image_gs, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, 2)
    return image_bin
def image_bin_adaptive_gauss(image_gs):
    image_bin = cv2.adaptiveThreshold(image_gs, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 2)
    return image_bin
def blur(image):
    return cv2.GaussianBlur(image,(5,5),0)
def invert(image):
    return 255-image
def display_image(image, color= False):
    plt.figure()
    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')
    plt.show()

#org_image = load_image("test_images/staff_lines.jpg")
#org_image = load_image("test_images/two-octaves.jpg")
org_image = load_image("test_images/staff-with-notes.jpg")
display_image(org_image, True)

print("Grayscale Image")
image = image_gray(org_image)
display_image(image)

print("Global Binarization")
img_bin = image_bin(image)
display_image(img_bin)
print("Otsu's Binarization")
img_otsu = image_bin_otsu(image)
display_image(img_otsu)
print("Adaptive Binarization")
img_adp = image_bin_adaptive(image)
display_image(img_adp)
print("Adaptive Gauss Binarization")
img_gauss = image_bin_adaptive_gauss(image)
display_image(img_gauss)

print("Blurred Image")
img_blur = blur(image)
display_image(img_blur)
print("Global Binarization")
img_bin = image_bin(img_blur)
display_image(img_bin)
print("Otsu's Binarization")
img_otsu = image_bin_otsu(img_blur)
display_image(img_otsu)
print("Adaptive Binarization")
img_adp = image_bin_adaptive(img_blur)
display_image(img_adp)
print("Adaptive Gauss Binarization")
img_gauss = image_bin_adaptive_gauss(img_blur)
display_image(img_gauss)



