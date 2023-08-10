import glob2

u1_directory = 'images/u1/'
u2_directory = 'images/u2/'
u3_directory = 'images/u3/'

u1_files = glob2.glob(u1_directory+'/**/*.jpg')
u2_files = glob2.glob(u2_directory+'/**/*.jpg')
u3_files = glob2.glob(u3_directory+'/**/*.jpg')

print 'User 1: {0} images'.format(len(u1_files))
print 'User 2: {0} images'.format(len(u2_files))
print 'User 3: {0} images'.format(len(u3_files))
print 'TOTAL: {0} images to PROCESS!'.format(len(u1_files)+len(u2_files)+len(u2_files))

images_directory = 'images'
image_files = glob2.glob(images_directory+'/**/*.jpg')
 
print image_files[0:10]

get_ipython().run_cell_magic('writefile', 'process_yolo.py', '\nimport multiprocessing\nimport subprocess\nimport os.path as path\nfrom multiprocessing import Pool\nimport sys\nimport glob2\n\ndef work(image_file):\n\n    if path.exists(image_file[:-3]+\'txt\'):\n        pass\n    else:\n        print image_file\n        output = subprocess.check_output([\'./darknet/darknet\', \'detector\', \'test\', \'cfg/coco.data\', \'cfg/yolo.cfg\', \'yolo.weights\',image_file,\'-thresh\',\'0.1\'])\n        f = open(image_file[:-3]+\'txt\',"w")\n        f.write(output)\n        f.close()\n        \nif __name__ == \'__main__\':\n\n    if len(sys.argv) == 2:\n        \n        image_directory = sys.argv[1]\n        image_files = glob2.glob(image_directory+\'/**/*.jpg\')\n        \n        for image_file in image_files:\n            work(image_file)\n        \n    else:\n        "Please enter the image path"')

import subprocess

get_ipython().run_cell_magic('timeit', '', "image_file = 'b00000985_21i79q_20150615_175218e.jpg'\noutput = subprocess.check_output(['./darknet', 'detector', 'test', 'cfg/coco.data', 'cfg/yolo.cfg', 'yolo.weights',image_file,'-thresh','0.1'])")

print(output)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img=mpimg.imread(image_file)
plt.figure(figsize=(12,16))
imgplot = plt.imshow(img)

img=mpimg.imread('predictions.png')
plt.figure(figsize=(12,16))
imgplot = plt.imshow(img)

get_ipython().magic('mkdir data/raw_yolo_u1/raw_data/')

get_ipython().system('mv -v images/u1/2015-03-09/*.txt data/raw_yolo_u1/raw_data/')

