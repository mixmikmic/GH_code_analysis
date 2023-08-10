import sys
import math
import numpy as np
import os
import time
from PIL import Image
from matplotlib import pyplot
import cv2
from datetime import datetime
from pynq import Xlnk
from pynq import Overlay
from preprocessing import Agent
from preprocessing import BATCH_SIZE
from preprocessing import get_image_path

team = 'pynquser'
agent = Agent(team)

OVERLAY_PATH = '/home/xilinx/jupyter_notebooks/dac_2018/'                 'overlay/pynquser/dac_contest.bit'
overlay = Overlay(OVERLAY_PATH)
dma = overlay.axi_dma_0

interval_time = 0
image_path = get_image_path('1.jpg')

original_image = Image.open(image_path)
original_array = np.array(original_image)
original_image.close()

pyplot.imshow(original_array, interpolation='nearest')
pyplot.show()

old_width, old_height = original_image.size
print("Original image size: {}x{} pixels.".format(old_height, old_width))

new_width, new_height = int(old_width/2), int(old_height/2)
original_image = Image.open(image_path)
resized_image = original_image.resize((new_width, new_height), 
                                      Image.ANTIALIAS)
resized_array = np.array(resized_image)
original_image.close()

pyplot.imshow(resized_array, interpolation='nearest')
pyplot.show()

width, height = resized_image.size
print("Resized image size: {}x{} pixels.".format(height, width))

xlnk = Xlnk()
in_buffer = xlnk.cma_array(shape=(263, 358, 3), dtype=np.uint8)
out_buffer = xlnk.cma_array(shape=(263, 358, 3), dtype=np.uint8)

interval_time = 0

image_path = get_image_path('0.jpg')
bgr_array = cv2.imread(image_path)
rgb_array = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2RGB)
in_buffer[:] = rgb_array

pyplot.imshow(in_buffer)
pyplot.show()

def dma_transfer():
    dma.sendchannel.transfer(in_buffer)
    dma.recvchannel.transfer(out_buffer)
    dma.sendchannel.wait()
    dma.recvchannel.wait()

dma_transfer()

pyplot.imshow(out_buffer)
pyplot.show()

interval_time = 0
total_time = 0
total_num_img = len(agent.img_list)
result = list()
agent.reset_batch_count()

for i in range(math.ceil(total_num_img/BATCH_SIZE)):
    # get a batch from agent
    batch = agent.send(interval_time, agent.img_batch)

    # choose a single image from the batch
    first_image = sorted(batch)[0]

    # timer start when PS reading image
    start = time.time()
    bgr_array = cv2.imread(get_image_path(first_image))
    rgb_array = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2RGB)
    in_buffer[:] = rgb_array
    dma_transfer()

    # timer stop after PS has received image
    end = time.time()
    t = end - start
    print('Processing time: {} seconds.'.format(t))
    total_time += t
    result.append(str(out_buffer))

agent.write(total_time, total_num_img, team)

with open(agent.coord_team + '/{}.txt'.format(team), 'w+') as fcoord:
    for element in result:
        fcoord.write(element)
        fcoord.write('\n')
print("Coordinate results written successfully.")

result_rectangle =  [[0,358,0,263],[0,1350,0,707]]

agent.save_results_xml(result_rectangle)
print("XML results written successfully.")

xlnk.xlnk_reset()

