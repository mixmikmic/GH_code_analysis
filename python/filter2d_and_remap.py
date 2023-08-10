from pynq.lib.video import *
from pynq_computervision import BareHDMIOverlay
base = BareHDMIOverlay("/opt/python3.6/lib/python3.6/site-packages/"
                       "pynq_computervision/overlays/computer_vision/"
                       "xv2Filter2DRemap.bit")
from pynq import Xlnk
mem_manager = Xlnk()
import pynq_computervision.overlays.computer_vision.xv2Filter2DRemap as xv2

hdmi_in = base.video.hdmi_in
hdmi_out = base.video.hdmi_out

hdmi_in.configure(PIXEL_GRAY)
hdmi_out.configure(hdmi_in.mode)

hdmi_in.start()
hdmi_out.start()

mymode = hdmi_in.mode
print("My mode: "+str(mymode))

height = hdmi_in.mode.height
width = hdmi_in.mode.width
bpp = hdmi_in.mode.bits_per_pixel

import numpy as np
import cv2

def makeMapCircleZoom(width, height, cx, cy, radius, zoom):
    mapY, mapX = np.indices((height,width),dtype=np.float32)
    
    for (j,i),x in np.ndenumerate(mapX[cy-radius:cy+radius,
                                       cx-radius:cx+radius]):
        x = i - radius
        y = j - radius
        i += cx-radius
        j += cy-radius
        mapX[(j,i)] = (cx + x/zoom) if (np.sqrt(x*x+y*y)<radius) else i
        mapY[(j,i)] = (cy + y/zoom) if (np.sqrt(x*x+y*y)<radius) else j

    return(mapX,mapY)

import numpy as np
import cv2

map1, map2 = makeMapCircleZoom(width,height,1200,540,60,2.0)

numframes = 12

start = time.time()
for _ in range(numframes):
    inframe = hdmi_in.readframe()
    outframe = hdmi_out.newframe()
    cv2.remap(inframe, map1, map2, cv2.INTER_LINEAR, dst=outframe)
    inframe.freebuffer()
    hdmi_out.writeframe(outframe)
end = time.time()

print("Frames per second:  " + str(numframes / (end - start)))

import PIL.Image

image = PIL.Image.fromarray(inframe)
image

import PIL.Image

image = PIL.Image.fromarray(outframe)
image

from ipywidgets import interact, interactive, fixed, interact_manual
from ipywidgets import IntSlider, FloatSlider
import ipywidgets as widgets

var_changed_g = 0
cx_g = width/2+240
cy_g = height/2
radius_g = 60
zoom_g = 2.0

def makeMapCircleZoomAndUpdate(width, height, cx, cy, radius, zoom):
    global var_changed_g
    global cx_g
    global cy_g
    global radius_g
    global zoom_g
    #print(var_changed_g,cx,cy,radius,zoom)
    if var_changed_g == 0:
        cx_g = cx
        cy_g = cy
        radius_g = radius
        zoom_g = zoom
        var_changed_g = 1
    #print(cx_g,cy_g,radius_g,zoom_g)
    
width_widget  = width;
height_widget = height;
cx_widget     = IntSlider(min=0,max=width-1, step=1, value=cx_g,
                          continuous_update=False)
cy_widget     = IntSlider(min=0,max=height-1, step=1, value=cy_g,
                          continuous_update=False)
radius_widget = IntSlider(min=1,max=100,step=1,value=radius_g,
                          continuous_update=False)
zoom_widget   = FloatSlider(min=0.1,max=4.0,step=0.1,value=zoom_g,
                            continuous_update=False)

interact(makeMapCircleZoomAndUpdate, width=fixed(width_widget), 
         height=fixed(height_widget), cx=cx_widget,cy=cy_widget,
         radius=radius_widget,zoom=zoom_widget)

def loop_hw_app():
    global var_changed_g

    map1, map2 = makeMapCircleZoom(width,height,cx_g,cy_g,radius_g,zoom_g)

    xFmap1 = mem_manager.cma_array((height,width),np.float32)
    xFmap2 = mem_manager.cma_array((height,width),np.float32)

    xFmap1[:] = map1[:]
    xFmap2[:] = map2[:]

    var_changed_g == 0
    numframes = 500

    start=time.time()
    for _ in range(numframes):
        inframe = hdmi_in.readframe()
        outframe = hdmi_out.newframe()
        if var_changed_g == 1:
            map1, map2 = makeMapCircleZoom(width, height, cx_g, cy_g, 
                                           radius_g, zoom_g)
            xFmap1[:] = map1[:]
            xFmap2[:] = map2[:]
            var_changed_g = 0
        xv2.remap(inframe, xFmap1, xFmap2, cv2.INTER_LINEAR, dst=outframe)
        inframe.freebuffer()
        hdmi_out.writeframe(outframe)
    end=time.time()

    print("Frames per second:  " + str(numframes / (end - start)))

from threading import Thread

t = Thread(target=loop_hw_app, )
t.start()

import PIL.Image

image = PIL.Image.fromarray(outframe)
image

buf =np.ones((height,width),np.uint8)
kernel = np.array([[0.0, 1.0, 0],[1.0,-4,1.0],[0,1.0,0.0]],np.float32)

map1, map2 = makeMapCircleZoom(width,height,1200,540,60,2.0)

xFbuf  = mem_manager.cma_array((height,width),np.uint8)
xFmap1 = mem_manager.cma_array((height,width),np.float32)
xFmap2 = mem_manager.cma_array((height,width),np.float32)

xFmap1[:] = map1[:]
xFmap2[:] = map2[:]

numframes = 220

start=time.time()
for _ in range(numframes):
    inframe = hdmi_in.readframe()
    outframe = hdmi_out.newframe()
    xv2.filter2D(inframe, -1, kernel, xFbuf, (-1,-1), 0.0, 
                 borderType=cv2.BORDER_CONSTANT)
    xv2.remap(xFbuf, xFmap1, xFmap2, cv2.INTER_LINEAR, dst=outframe)
    inframe.freebuffer()
    hdmi_out.writeframe(outframe)
end=time.time()

print("Frames per second:  " + str(numframes / (end - start)))

import PIL.Image

image = PIL.Image.fromarray(outframe)
image

hdmi_out.close()
hdmi_in.close()



