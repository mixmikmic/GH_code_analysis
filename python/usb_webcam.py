from PIL import Image as PIL_Image

orig_img_path = '/home/xilinx/jupyter_notebooks/examples/data/webcam.jpg'
get_ipython().system('fswebcam  --no-banner --save {orig_img_path} -d /dev/video0 2> /dev/null')

img = PIL_Image.open(orig_img_path)
img

bw_img = img.convert("L")
bw_img

rot_img = img.rotate(45)
rot_img

