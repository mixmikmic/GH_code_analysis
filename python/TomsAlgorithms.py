
from PIL import Image, ImageChops, ImageStat, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import glob


def feqencyextractor(testimg, pixelfrequency, imgtype=1, max_pixel=255):
    nimg2 = Image.new('RGB', testimg.size, (127, 127, 127))
    nimg3 = Image.new('RGB', testimg.size, (0, 0, 0))
    for k in [0, 90, 180, 270]:
        img = testimg.rotate(k)
        for i in range(20):
            shkimg = img.resize((testimg.size[0] - pixelfrequency,
                                 testimg.size[1]))
            shkimg = ImageChops.invert(shkimg)
            tmpimg = img.copy()
            tmpimg.paste(shkimg, (pixelfrequency, 0))
            img = ImageChops.blend(img, tmpimg, .5)
        nimg = img.crop((int(pixelfrequency * 20 / 2), 0,
                         testimg.size[0], testimg.size[1]))
        nimg = nimg.resize(testimg.size)
        nimg = nimg.rotate(360 - k)
        img = img.rotate(360 - k)
        nimg2 = ImageChops.add_modulo(nimg2, img)
        nimg3 = ImageChops.blend(nimg3, nimg, .5)
    if imgtype == 1:

        return nimg3
    
    elif imgtype == 2:

        return nimg2
    
    elif imgtype == 3:
        (r, g, b) = imgtouxmRGB(nimg3)
        nr = []
        for i in r:
            nnr = []
            for j in i:
                if j == 127:
                    nnr.append(max_pixel)
                else:
                    nnr.append(0)
            nr.append(nnr)
        ng = []
        for i in g:
            nng = []
            for j in i:
                if j == 127:
                    nng.append(max_pixel)
                else:
                    nng.append(0)
            ng.append(nng)
        nb = []
        for i in b:
            nnb = []
            for j in i:
                if j == 127:
                    nnb.append(max_pixel)
                else:
                    nnb.append(0)
            nb.append(nnb)
        nr = np.array(nr)
        ng = np.array(ng)
        nb = np.array(nb)
        r2 = imguxret(nr)
        g2 = imguxret(ng)
        b2 = imguxret(nb)

        return (r2, g2, b2)


def feqencyextractor_mirror(testimg, pixelfrequency, imgtype=1, max_pixel=255):
    nimg2 = Image.new('RGB', testimg.size, (127, 127, 127))
    nimg3 = Image.new('RGB', testimg.size, (0, 0, 0))
    for k in [0, 90, 180, 270]:
        img = testimg.rotate(k)
        for i in range(20):
            shkimg = img.resize((testimg.size[0] - pixelfrequency,
                                 testimg.size[1]))
            shkimg = ImageChops.invert(shkimg)
            shkimg = shkimg.transpose(Image.FLIP_LEFT_RIGHT)
            tmpimg = img.copy()
            tmpimg.paste(shkimg, (pixelfrequency, 0))
            img = ImageChops.blend(img, tmpimg, .5)
        nimg = img.crop((int(pixelfrequency * 20 / 2), 0,
                         testimg.size[0], testimg.size[1]))
        nimg = nimg.resize(testimg.size)
        nimg = nimg.rotate(360 - k)
        img = img.rotate(360 - k)
        nimg2 = ImageChops.add_modulo(nimg2, img)
        nimg3 = ImageChops.blend(nimg3, nimg, .5)
    if imgtype == 1:

        return nimg3
    
    elif imgtype == 2:

        return nimg2
    
    elif imgtype == 3:

        (r, g, b) = imgtouxmRGB(nimg3)
        nr = []
        for i in r:
            nnr = []
            for j in i:
                if j == 127:
                    nnr.append(max_pixel)
                else:
                    nnr.append(0)
            nr.append(nnr)
        ng = []
        for i in g:
            nng = []
            for j in i:
                if j == 127:
                    nng.append(max_pixel)
                else:
                    nng.append(0)
            ng.append(nng)
        nb = []
        for i in b:
            nnb = []
            for j in i:
                if j == 127:
                    nnb.append(max_pixel)
                else:
                    nnb.append(0)
            nb.append(nnb)
        nr = np.array(nr)
        ng = np.array(ng)
        nb = np.array(nb)
        r2 = imguxret(nr)
        g2 = imguxret(ng)
        b2 = imguxret(nb)

        return (r2, g2, b2)


def feqencyextractor_gro(testimg, pixelfrequency, imgtype=1, max_pixel=255):
    nimg2 = Image.new('RGB', testimg.size, (127, 127, 127))
    nimg3 = Image.new('RGB', testimg.size, (0, 0, 0))
    for k in [0, 90, 180, 270]:
        img = testimg.rotate(k)
        for i in range(20):
            shkimg = img.resize((testimg.size[0] + pixelfrequency,
                                 testimg.size[1]))
            shkimg = ImageChops.invert(shkimg)
            tmpimg = img.copy()
            tmpimg.paste(shkimg, (-pixelfrequency, 0))
            img = ImageChops.blend(img, tmpimg, .5)
        nimg = img.crop((int(pixelfrequency * 20 / 2), 0,
                         testimg.size[0], testimg.size[1]))
        nimg = nimg.resize(testimg.size)
        nimg = nimg.rotate(360 - k)
        img = img.rotate(360 - k)
        nimg2 = ImageChops.add_modulo(nimg2, img)
        nimg3 = ImageChops.blend(nimg3, nimg, .5)
    if imgtype == 1:

        return nimg3
    
    elif imgtype == 2:

        return nimg2
    
    elif imgtype == 3:

        (r, g, b) = imgtouxmRGB(nimg3)
        nr = []
        for i in r:
            nnr = []
            for j in i:
                if j == 127:
                    nnr.append(max_pixel)
                else:
                    nnr.append(0)
            nr.append(nnr)
        ng = []
        for i in g:
            nng = []
            for j in i:
                if j == 127:
                    nng.append(max_pixel)
                else:
                    nng.append(0)
            ng.append(nng)
        nb = []
        for i in b:
            nnb = []
            for j in i:
                if j == 127:
                    nnb.append(max_pixel)
                else:
                    nnb.append(0)
            nb.append(nnb)
        nr = np.array(nr)
        ng = np.array(ng)
        nb = np.array(nb)
        r2 = imguxret(nr)
        g2 = imguxret(ng)
        b2 = imguxret(nb)

        return (r2, g2, b2)


def feqencyextractor_diag(testimg, pixelfrequency, imgtype=1, max_pixel=255):
    nimg2 = Image.new('RGB', testimg.size, (127, 127, 127))
    nimg3 = Image.new('RGB', testimg.size, (0, 0, 0))
    for k in [0, 90, 180, 270]:
        img = testimg.rotate(k)
        for i in range(20):
            shkimg = img.resize((testimg.size[0] - pixelfrequency,
                                 testimg.size[1] - pixelfrequency))
            shkimg = ImageChops.invert(shkimg)
            tmpimg = img.copy()
            tmpimg.paste(shkimg, (pixelfrequency, pixelfrequency))
            img = ImageChops.blend(img, tmpimg, .5)
        img = img.rotate(360 - k)
        nimg2 = ImageChops.add_modulo(nimg2, img)
        nimg3 = ImageChops.blend(nimg3, img, .5)
    if imgtype == 1:

        return nimg3
    
    elif imgtype == 2:

        return nimg2
    
    elif imgtype == 3:

        (r, g, b) = imgtouxmRGB(nimg3)
        nr = []
        for i in r:
            nnr = []
            for j in i:
                if j == 127:
                    nnr.append(max_pixel)
                else:
                    nnr.append(0)
            nr.append(nnr)
        ng = []
        for i in g:
            nng = []
            for j in i:
                if j == 127:
                    nng.append(max_pixel)
                else:
                    nng.append(0)
            ng.append(nng)
        nb = []
        for i in b:
            nnb = []
            for j in i:
                if j == 127:
                    nnb.append(max_pixel)
                else:
                    nnb.append(0)
            nb.append(nnb)
        nr = np.array(nr)
        ng = np.array(ng)
        nb = np.array(nb)
        r2 = imguxret(nr)
        g2 = imguxret(ng)
        b2 = imguxret(nb)

        return (r2, g2, b2)


def feqencyextractor_diag_mirror(testimg, pixelfrequency, imgtype=1, max_pixel=255):
    nimg2 = Image.new('RGB', testimg.size, (127, 127, 127))
    nimg3 = Image.new('RGB', testimg.size, (0, 0, 0))
    for k in [0, 90, 180, 270]:
        img = testimg.rotate(k)
        for i in range(20):
            shkimg = img.resize((testimg.size[0] - pixelfrequency,
                                 testimg.size[1] - pixelfrequency))
            shkimg = ImageChops.invert(shkimg)
            shkimg = shkimg.transpose(Image.FLIP_LEFT_RIGHT)
            shkimg = shkimg.transpose(Image.FLIP_TOP_BOTTOM)
            tmpimg = img.copy()
            tmpimg.paste(shkimg, (pixelfrequency, pixelfrequency))
            img = ImageChops.blend(img, tmpimg, .5)
        img = img.rotate(360 - k)
        nimg2 = ImageChops.add_modulo(nimg2, img)
        nimg3 = ImageChops.blend(nimg3, img, .5)
    if imgtype == 1:

        return nimg3
    
    elif imgtype == 2:

        return nimg2
    
    elif imgtype == 3:

        (r, g, b) = imgtouxmRGB(nimg3)
        nr = []
        for i in r:
            nnr = []
            for j in i:
                if j == 127:
                    nnr.append(max_pixel)
                else:
                    nnr.append(0)
            nr.append(nnr)
        ng = []
        for i in g:
            nng = []
            for j in i:
                if j == 127:
                    nng.append(max_pixel)
                else:
                    nng.append(0)
            ng.append(nng)
        nb = []
        for i in b:
            nnb = []
            for j in i:
                if j == 127:
                    nnb.append(max_pixel)
                else:
                    nnb.append(0)
            nb.append(nnb)
        nr = np.array(nr)
        ng = np.array(ng)
        nb = np.array(nb)
        r2 = imguxret(nr)
        g2 = imguxret(ng)
        b2 = imguxret(nb)

        return (r2, g2, b2)


def feqencyextractor_diag_gro(testimg, pixelfrequency, imgtype=1, max_pixel=255):
    nimg2 = Image.new('RGB', testimg.size, (127, 127, 127))
    nimg3 = Image.new('RGB', testimg.size, (0, 0, 0))
    for k in [0, 90, 180, 270]:
        img = testimg.rotate(k)
        for i in range(20):
            shkimg = img.resize((testimg.size[0] + pixelfrequency,
                                 testimg.size[1] + pixelfrequency))
            shkimg = ImageChops.invert(shkimg)
            tmpimg = img.copy()
            tmpimg.paste(shkimg, (-pixelfrequency, -pixelfrequency))
            img = ImageChops.blend(img, tmpimg, .5)
        img = img.rotate(360 - k)
        nimg2 = ImageChops.add_modulo(nimg2, img)
        nimg3 = ImageChops.blend(nimg3, img, .5)
    if imgtype == 1:

        return nimg3
    
    elif imgtype == 2:

        return nimg2
    
    elif imgtype == 3:

        (r, g, b) = imgtouxmRGB(nimg3)
        nr = []
        for i in r:
            nnr = []
            for j in i:
                if j == 127:
                    nnr.append(max_pixel)
                else:
                    nnr.append(0)
            nr.append(nnr)
        ng = []
        for i in g:
            nng = []
            for j in i:
                if j == 127:
                    nng.append(max_pixel)
                else:
                    nng.append(0)
            ng.append(nng)
        nb = []
        for i in b:
            nnb = []
            for j in i:
                if j == 127:
                    nnb.append(max_pixel)
                else:
                    nnb.append(0)
            nb.append(nnb)
        nr = np.array(nr)
        ng = np.array(ng)
        nb = np.array(nb)
        r2 = imguxret(nr)
        g2 = imguxret(ng)
        b2 = imguxret(nb)

        return (r2, g2, b2)


def feqencyextractor_diag_mirror_gro(testimg, pixelfrequency, imgtype=1, max_pixel=255):
    nimg2 = Image.new('RGB', testimg.size, (127, 127, 127))
    nimg3 = Image.new('RGB', testimg.size, (0, 0, 0))
    for k in [0, 90, 180, 270]:
        img = testimg.rotate(k)
        for i in range(20):
            shkimg = img.resize((testimg.size[0] + pixelfrequency,
                                 testimg.size[1] + pixelfrequency))
            shkimg = ImageChops.invert(shkimg)
            shkimg = shkimg.transpose(Image.FLIP_LEFT_RIGHT)
            shkimg = shkimg.transpose(Image.FLIP_TOP_BOTTOM)
            tmpimg = img.copy()
            tmpimg.paste(shkimg, (-pixelfrequency, -pixelfrequency))
            img = ImageChops.blend(img, tmpimg, .5)
        img = img.rotate(360 - k)
        nimg2 = ImageChops.add_modulo(nimg2, img)
        nimg3 = ImageChops.blend(nimg3, img, .5)
    if imgtype == 1:

        return nimg3
    
    elif imgtype == 2:

        return nimg2
    
    elif imgtype == 3:

        (r, g, b) = imgtouxmRGB(nimg3)
        nr = []
        for i in r:
            nnr = []
            for j in i:
                if j == 127:
                    nnr.append(max_pixel)
                else:
                    nnr.append(0)
            nr.append(nnr)
        ng = []
        for i in g:
            nng = []
            for j in i:
                if j == 127:
                    nng.append(max_pixel)
                else:
                    nng.append(0)
            ng.append(nng)
        nb = []
        for i in b:
            nnb = []
            for j in i:
                if j == 127:
                    nnb.append(max_pixel)
                else:
                    nnb.append(0)
            nb.append(nnb)
        nr = np.array(nr)
        ng = np.array(ng)
        nb = np.array(nb)
        r2 = imguxret(nr)
        g2 = imguxret(ng)
        b2 = imguxret(nb)

        return (r2, g2, b2)


def feqencyextractor_rot(testimg, pixelfrequency, imgtype=1, max_pixel=255, rotation=2):
    nimg2 = Image.new('RGB', testimg.size, (127, 127, 127))
    nimg3 = Image.new('RGB', testimg.size, (0, 0, 0))
    for k in [0, 90, 180, 270]:
        img = testimg.rotate(k)
        for i in range(20):
            img = shrinky_rot(img, pixelfrequency, rotation)
        img = img.rotate(360 - k)
        nimg2 = ImageChops.add_modulo(nimg2, img)
        nimg3 = ImageChops.blend(nimg3, img, .5)
    if imgtype == 1:

        return nimg3
    
    elif imgtype == 2:

        return nimg2
    
    elif imgtype == 3:

        (r, g, b) = imgtouxmRGB(nimg3)
        nr = []
        for i in r:
            nnr = []
            for j in i:
                if j == 127:
                    nnr.append(max_pixel)
                else:
                    nnr.append(0)
            nr.append(nnr)
        ng = []
        for i in g:
            nng = []
            for j in i:
                if j == 127:
                    nng.append(max_pixel)
                else:
                    nng.append(0)
            ng.append(nng)
        nb = []
        for i in b:
            nnb = []
            for j in i:
                if j == 127:
                    nnb.append(max_pixel)
                else:
                    nnb.append(0)
            nb.append(nnb)
        nr = np.array(nr)
        ng = np.array(ng)
        nb = np.array(nb)
        r2 = imguxret(nr)
        g2 = imguxret(ng)
        b2 = imguxret(nb)

        return (r2, g2, b2)


def feqencyextractor_rot_mirror(testimg, pixelfrequency, imgtype=1, max_pixel=255, rotation=2):
    nimg2 = Image.new('RGB', testimg.size, (127, 127, 127))
    nimg3 = Image.new('RGB', testimg.size, (0, 0, 0))
    for k in [0, 90, 180, 270]:
        img = testimg.rotate(k)
        for i in range(20):
            img = shrinky_rot_mirror(img, pixelfrequency, rotation)
        img = img.rotate(360 - k)
        nimg2 = ImageChops.add_modulo(nimg2, img)
        nimg3 = ImageChops.blend(nimg3, img, .5)
    if imgtype == 1:

        return nimg3
    
    elif imgtype == 2:

        return nimg2
    
    elif imgtype == 3:

        (r, g, b) = imgtouxmRGB(nimg3)
        nr = []
        for i in r:
            nnr = []
            for j in i:
                if j == 127:
                    nnr.append(max_pixel)
                else:
                    nnr.append(0)
            nr.append(nnr)
        ng = []
        for i in g:
            nng = []
            for j in i:
                if j == 127:
                    nng.append(max_pixel)
                else:
                    nng.append(0)
            ng.append(nng)
        nb = []
        for i in b:
            nnb = []
            for j in i:
                if j == 127:
                    nnb.append(max_pixel)
                else:
                    nnb.append(0)
            nb.append(nnb)
        nr = np.array(nr)
        ng = np.array(ng)
        nb = np.array(nb)
        r2 = imguxret(nr)
        g2 = imguxret(ng)
        b2 = imguxret(nb)

        return (r2, g2, b2)


def feqencyextractor_rot_gro(testimg, pixelfrequency, imgtype=1, max_pixel=255, rotation=2):
    nimg2 = Image.new('RGB', testimg.size, (127, 127, 127))
    nimg3 = Image.new('RGB', testimg.size, (0, 0, 0))
    for k in [0, 90, 180, 270]:
        img = testimg.rotate(k)
        for i in range(20):
            img = shrinky_rot_gro(img, pixelfrequency, rotation)
        img = img.rotate(360 - k)
        nimg2 = ImageChops.add_modulo(nimg2, img)
        nimg3 = ImageChops.blend(nimg3, img, .5)
    if imgtype == 1:

        return nimg3
    
    elif imgtype == 2:

        return nimg2
    
    elif imgtype == 3:

        (r, g, b) = imgtouxmRGB(nimg3)
        nr = []
        for i in r:
            nnr = []
            for j in i:
                if j == 127:
                    nnr.append(max_pixel)
                else:
                    nnr.append(0)
            nr.append(nnr)
        ng = []
        for i in g:
            nng = []
            for j in i:
                if j == 127:
                    nng.append(max_pixel)
                else:
                    nng.append(0)
            ng.append(nng)
        nb = []
        for i in b:
            nnb = []
            for j in i:
                if j == 127:
                    nnb.append(max_pixel)
                else:
                    nnb.append(0)
            nb.append(nnb)
        nr = np.array(nr)
        ng = np.array(ng)
        nb = np.array(nb)
        r2 = imguxret(nr)
        g2 = imguxret(ng)
        b2 = imguxret(nb)

        return (r2, g2, b2)


def feqencyextractor_rot_mirror_gro(testimg, pixelfrequency, imgtype=1, max_pixel=255, rotation=2):
    nimg2 = Image.new('RGB', testimg.size, (127, 127, 127))
    nimg3 = Image.new('RGB', testimg.size, (0, 0, 0))
    for k in [0, 90, 180, 270]:
        img = testimg.rotate(k)
        for i in range(20):
            img = shrinky_rot_mirror_gro(img, pixelfrequency, rotation)
        img = img.rotate(360 - k)
        nimg2 = ImageChops.add_modulo(nimg2, img)
        nimg3 = ImageChops.blend(nimg3, img, .5)
    if imgtype == 1:

        return nimg3
    
    elif imgtype == 2:

        return nimg2
    
    elif imgtype == 3:

        (r, g, b) = imgtouxmRGB(nimg3)
        nr = []
        for i in r:
            nnr = []
            for j in i:
                if j == 127:
                    nnr.append(max_pixel)
                else:
                    nnr.append(0)
            nr.append(nnr)
        ng = []
        for i in g:
            nng = []
            for j in i:
                if j == 127:
                    nng.append(max_pixel)
                else:
                    nng.append(0)
            ng.append(nng)
        nb = []
        for i in b:
            nnb = []
            for j in i:
                if j == 127:
                    nnb.append(max_pixel)
                else:
                    nnb.append(0)
            nb.append(nnb)
        nr = np.array(nr)
        ng = np.array(ng)
        nb = np.array(nb)
        r2 = imguxret(nr)
        g2 = imguxret(ng)
        b2 = imguxret(nb)

        return (r2, g2, b2)


def feqencyextractor_skew(testimg, pixelfrequency, imgtype=1, max_pixel=255):
    nimg2 = Image.new('RGB', testimg.size, (127, 127, 127))
    nimg3 = Image.new('RGB', testimg.size, (0, 0, 0))
    for k in [0, 90, 180, 270]:
        img = testimg.rotate(k)
        imgtemp = img.copy()
        for i in range(20):
            img = skew_up(img, pixelfrequency)
        for i in range(20):
            imgtemp = skew_down(imgtemp, pixelfrequency)
        img = ImageChops.blend(img, imgtemp, .5)
        img = img.rotate(360 - k)
        nimg2 = ImageChops.add_modulo(nimg2, img)
        nimg3 = ImageChops.blend(nimg3, img, .5)
    if imgtype == 1:

        return nimg3
    
    elif imgtype == 2:

        return nimg2
    
    elif imgtype == 3:

        (r, g, b) = imgtouxmRGB(nimg3)
        nr = []
        for i in r:
            nnr = []
            for j in i:
                if j == 127:
                    nnr.append(max_pixel)
                else:
                    nnr.append(0)
            nr.append(nnr)
        ng = []
        for i in g:
            nng = []
            for j in i:
                if j == 127:
                    nng.append(max_pixel)
                else:
                    nng.append(0)
            ng.append(nng)
        nb = []
        for i in b:
            nnb = []
            for j in i:
                if j == 127:
                    nnb.append(max_pixel)
                else:
                    nnb.append(0)
            nb.append(nnb)
        nr = np.array(nr)
        ng = np.array(ng)
        nb = np.array(nb)
        r2 = imguxret(nr)
        g2 = imguxret(ng)
        b2 = imguxret(nb)

        return (r2, g2, b2)


def feqencyextractor_skew_mirror(testimg, pixelfrequency, imgtype=1, max_pixel=255):
    nimg2 = Image.new('RGB', testimg.size, (127, 127, 127))
    nimg3 = Image.new('RGB', testimg.size, (0, 0, 0))
    for k in [0, 90, 180, 270]:
        img = testimg.rotate(k)
        imgtemp = img.copy()
        for i in range(20):
            img = skew_up_mirror(img, pixelfrequency)
        for i in range(20):
            imgtemp = skew_down_mirror(imgtemp, pixelfrequency)
        img = ImageChops.blend(img, imgtemp, .5)
        img = img.rotate(360 - k)
        nimg2 = ImageChops.add_modulo(nimg2, img)
        nimg3 = ImageChops.blend(nimg3, img, .5)
    if imgtype == 1:

        return nimg3
    
    elif imgtype == 2:

        return nimg2
    
    elif imgtype == 3:

        (r, g, b) = imgtouxmRGB(nimg3)
        nr = []
        for i in r:
            nnr = []
            for j in i:
                if j == 127:
                    nnr.append(max_pixel)
                else:
                    nnr.append(0)
            nr.append(nnr)
        ng = []
        for i in g:
            nng = []
            for j in i:
                if j == 127:
                    nng.append(max_pixel)
                else:
                    nng.append(0)
            ng.append(nng)
        nb = []
        for i in b:
            nnb = []
            for j in i:
                if j == 127:
                    nnb.append(max_pixel)
                else:
                    nnb.append(0)
            nb.append(nnb)
        nr = np.array(nr)
        ng = np.array(ng)
        nb = np.array(nb)
        r2 = imguxret(nr)
        g2 = imguxret(ng)
        b2 = imguxret(nb)

        return (r2, g2, b2)


def feqencyextractor_skew_gro(testimg, pixelfrequency, imgtype=1, max_pixel=255):
    nimg2 = Image.new('RGB', testimg.size, (127, 127, 127))
    nimg3 = Image.new('RGB', testimg.size, (0, 0, 0))
    for k in [0, 90, 180, 270]:
        img = testimg.rotate(k)
        imgtemp = img.copy()
        for i in range(20):
            img = skew_up_gro(img, pixelfrequency)
        for i in range(20):
            imgtemp = skew_down_gro(imgtemp, pixelfrequency)
        img = ImageChops.blend(img, imgtemp, .5)
        img = img.rotate(360 - k)
        nimg2 = ImageChops.add_modulo(nimg2, img)
        nimg3 = ImageChops.blend(nimg3, img, .5)
    if imgtype == 1:

        return nimg3
    
    elif imgtype == 2:

        return nimg2
    
    elif imgtype == 3:

        (r, g, b) = imgtouxmRGB(nimg3)
        nr = []
        for i in r:
            nnr = []
            for j in i:
                if j == 127:
                    nnr.append(max_pixel)
                else:
                    nnr.append(0)
            nr.append(nnr)
        ng = []
        for i in g:
            nng = []
            for j in i:
                if j == 127:
                    nng.append(max_pixel)
                else:
                    nng.append(0)
            ng.append(nng)
        nb = []
        for i in b:
            nnb = []
            for j in i:
                if j == 127:
                    nnb.append(max_pixel)
                else:
                    nnb.append(0)
            nb.append(nnb)
        nr = np.array(nr)
        ng = np.array(ng)
        nb = np.array(nb)
        r2 = imguxret(nr)
        g2 = imguxret(ng)
        b2 = imguxret(nb)

        return (r2, g2, b2)


def multifold(img):
    half_vert = img.resize((int(img.size[0] / 2), img.size[1]))
    half_horz = img.resize((img.size[1], int(img.size[0] / 2)))
    quarter = img.resize((int(img.size[0] / 2), int(img.size[0] / 2)))
    im1 = Image.new('RGB', img.size)
    im2 = Image.new('RGB', img.size)
    im3 = Image.new('RGB', img.size)
    im1.paste(half_vert, (0, 0))
    im1.paste(half_vert, (int(img.size[0] / 2), 0))
    im2.paste(half_horz, (0, 0))
    im2.paste(half_horz, (0, int(img.size[0] / 2)))
    im3.paste(quarter, (0, 0))
    im3.paste(quarter, (int(img.size[0] / 2), 0))
    im3.paste(quarter, (0, int(img.size[0] / 2)))
    im3.paste(quarter, (int(img.size[0] / 2), int(img.size[0] / 2)))
    im1 = ImageChops.invert(im1)
    im2 = ImageChops.invert(im2)
    im3 = ImageChops.invert(im3)
    im1 = ImageChops.blend(img, im1, .4)
    im2 = ImageChops.blend(img, im2, .4)
    im3 = ImageChops.blend(img, im3, .4)
    im1 = ImageChops.blend(im2, im1, .4)
    im1 = ImageChops.blend(im3, im1, .4)

    return im1


def shrinky(img, step=2):
    im2 = img.copy()
    smaller = img.resize((img.size[0] - step * 2,
                          img.size[1] - step * 2))
    smaller = ImageChops.invert(smaller)
    im2.paste(smaller, (step, step))
    im2 = ImageChops.blend(img, im2, .5)

    return im2


def shrinky_rot(img, step=2, rotation_deg=1):
    degrees = [0, 180, 120, 90, 60]
    im2 = img.copy()
    smaller = img.resize((img.size[0] - step * 2,
                          img.size[1] - step * 2))
    smaller = smaller.rotate(rotation_deg)
    smaller = ImageChops.invert(smaller)
    im2.paste(smaller, (step, step))
    im2 = ImageChops.blend(img, im2, .5)

    return im2


def shrinky_rot_mirror(img, step=2, rotation_deg=1):
    degrees = [0, 180, 120, 90, 60]
    im2 = img.copy()
    smaller = img.resize((img.size[0] - step * 2,
                          img.size[1] - step * 2))
    smaller = smaller.rotate(rotation_deg)
    smaller = ImageChops.invert(smaller)
    smaller = smaller.transpose(Image.FLIP_LEFT_RIGHT)
    smaller = smaller.transpose(Image.FLIP_TOP_BOTTOM)
    im2.paste(smaller, (step, step))
    im2 = ImageChops.blend(img, im2, .5)

    return im2


def shrinky_rot_gro(img, step=2, rotation_deg=1):
    degrees = [0, 180, 120, 90, 60]
    im2 = img.copy()
    smaller = img.resize((img.size[0] + step * 2,
                          img.size[1] + step * 2))
    smaller = smaller.rotate(degrees[rotation_deg])
    smaller = ImageChops.invert(smaller)
    im2.paste(smaller, (-step, -step))
    im2 = ImageChops.blend(img, im2, .5)

    return im2


def shrinky_rot_mirror_gro(img, step=2, rotation_deg=1):
    degrees = [0, 180, 120, 90, 60]
    im2 = img.copy()
    smaller = img.resize((img.size[0] + step * 2,
                          img.size[1] + step * 2))
    smaller = smaller.rotate(degrees[rotation_deg])
    smaller = ImageChops.invert(smaller)
    smaller = smaller.transpose(Image.FLIP_LEFT_RIGHT)
    smaller = smaller.transpose(Image.FLIP_TOP_BOTTOM)
    im2.paste(smaller, (-step, -step))
    im2 = ImageChops.blend(img, im2, .5)

    return im2


def skew_up(img, step=2.):
    im2 = ImageChops.invert(img.copy())
    small_steps = 0
    for y_row in range(img.size[1] - 1):
        if img.size[0] - small_steps > 1:
            img_chunk = img.crop((0, y_row, img.size[0], y_row + 1))
            img_chunk = img_chunk.resize((img_chunk.size[0] - int(small_steps),
                                          img_chunk.size[1]))
            img_chunk = ImageChops.invert(img_chunk)
            im2.paste(img_chunk, (int(small_steps), y_row))
        small_steps = small_steps + step
    im2 = ImageChops.blend(img, im2, .5)

    return im2


def skew_up_mirror(img, step=2.):
    im2 = ImageChops.invert(img.copy())
    small_steps = 0
    for y_row in range(img.size[1] - 1):
        if img.size[0] - small_steps > 1:
            img_chunk = img.crop((0, y_row, img.size[0], y_row + 1))
            img_chunk = img_chunk.resize((img_chunk.size[0] - int(small_steps),
                                          img_chunk.size[1]))
            img_chunk = ImageChops.invert(img_chunk)
            img_chunk = img_chunk.transpose(Image.FLIP_LEFT_RIGHT)
            im2.paste(img_chunk, (int(small_steps), y_row))
        small_steps = small_steps + step
    im2 = ImageChops.blend(img, im2, .5)

    return im2


def skew_up_gro(img, step=2.):
    im2 = ImageChops.invert(img.copy())
    small_steps = 0
    for y_row in range(img.size[1] - 1):
        if img.size[0] - small_steps > 1:
            img_chunk = img.crop((0, y_row, img.size[0], y_row + 1))
            img_chunk = img_chunk.resize((img_chunk.size[0] + int(small_steps),
                                          img_chunk.size[1]))
            img_chunk = ImageChops.invert(img_chunk)
            im2.paste(img_chunk, (-int(small_steps), y_row))
        small_steps = small_steps + step
    im2 = ImageChops.blend(img, im2, .5)

    return im2


def skew_down(img, step=2.):
    im2 = ImageChops.invert(img.copy())
    small_steps = 0
    for y_row in range(img.size[1] - 1, 0, -1):
        if img.size[0] - small_steps > 1:
            img_chunk = img.crop((0, y_row, img.size[0], y_row + 1))
            img_chunk = img_chunk.resize((img_chunk.size[0] - int(small_steps),
                                          img_chunk.size[1]))
            img_chunk = ImageChops.invert(img_chunk)
            im2.paste(img_chunk, (int(small_steps), y_row))
        small_steps = small_steps + step
    im2 = ImageChops.blend(img, im2, .5)

    return im2


def skew_down_mirror(img, step=2.):
    im2 = ImageChops.invert(img.copy())
    small_steps = 0
    for y_row in range(img.size[1] - 1, 0, -1):
        if img.size[0] - small_steps > 1:
            img_chunk = img.crop((0, y_row, img.size[0], y_row + 1))
            img_chunk = img_chunk.resize((img_chunk.size[0] - int(small_steps),
                                          img_chunk.size[1]))
            img_chunk = ImageChops.invert(img_chunk)
            img_chunk = img_chunk.transpose(Image.FLIP_LEFT_RIGHT)
            im2.paste(img_chunk, (int(small_steps), y_row))
        small_steps = small_steps + step
    im2 = ImageChops.blend(img, im2, .5)

    return im2


def skew_down_gro(img, step=2.):
    im2 = ImageChops.invert(img.copy())
    small_steps = 0
    for y_row in range(img.size[1] - 1, 0, -1):
        if img.size[0] - small_steps > 1:
            img_chunk = img.crop((0, y_row, img.size[0], y_row + 1))
            img_chunk = img_chunk.resize((img_chunk.size[0] + int(small_steps),
                                          img_chunk.size[1]))
            img_chunk = ImageChops.invert(img_chunk)
            im2.paste(img_chunk, (-int(small_steps), y_row))
        small_steps = small_steps + step
    im2 = ImageChops.blend(img, im2, .5)

    return im2


def unmultifold(img):
    
    half_vert1 = img.crop((0, 0, int(img.size[0] / 2), img.size[1]))
    half_vert1 = half_vert1.resize(img.size)
    half_vert2 = img.crop((int(img.size[1] / 2), 0, img.size[0],
                           img.size[1]))
    half_vert2 = half_vert2.resize(img.size)
    
    half_horz1 = img.crop((0, int(img.size[0] / 2), img.size[0],
                           img.size[1]))
    half_horz1 = half_horz1.resize(img.size)
    half_horz2 = img.crop((0, 0, img.size[0], int(img.size[0] / 2)))
    half_horz2 = half_horz2.resize(img.size)
    
    quarter1 = img.crop((int(img.size[0] / 2), int(img.size[0] / 2),
                         img.size[0], img.size[1]))
    quarter1 = quarter1.resize(img.size)
    
    quarter2 = img.crop((0, 0, int(img.size[0] / 2),
                         int(img.size[0] / 2)))
    quarter2 = quarter2.resize(img.size)
    
    quarter3 = img.crop((int(img.size[0] / 2), int(img.size[0] / 2),
                         img.size[0], img.size[1]))
    quarter3 = quarter3.resize(img.size)
    
    quarter4 = img.crop((int(img.size[0] / 2), int(img.size[0] / 2),
                         img.size[0], img.size[1]))
    quarter4 = quarter4.resize(img.size)

    half_vert1 = ImageChops.invert(half_vert1)
    half_vert2 = ImageChops.invert(half_vert2)
    half_horz1 = ImageChops.invert(half_horz1)
    half_horz2 = ImageChops.invert(half_horz2)

    quarter1 = ImageChops.invert(quarter1)
    quarter2 = ImageChops.invert(quarter2)
    quarter3 = ImageChops.invert(quarter3)
    quarter4 = ImageChops.invert(quarter4)
    
    half_vert1 = ImageChops.blend(img, half_vert1, .5)
    half_vert2 = ImageChops.blend(img, half_vert2, .5)
    half_horz1 = ImageChops.blend(img, half_horz1, .5)
    half_horz2 = ImageChops.blend(img, half_horz2, .5)

    quarter1 = ImageChops.blend(img, quarter1, .5)
    quarter2 = ImageChops.blend(img, quarter2, .5)
    quarter3 = ImageChops.blend(img, quarter3, .5)
    quarter4 = ImageChops.blend(img, quarter4, .5)
    
    im1 = ImageChops.blend(half_vert1, half_vert2, .5)
    im2 = ImageChops.blend(half_horz1, half_horz1, .5)
    im3a = ImageChops.blend(quarter1, quarter3, .5)
    im3b = ImageChops.blend(quarter2, quarter4, .5)
    im3 = ImageChops.blend(im3a, im3b, .5)
    im1 = ImageChops.blend(im2, im1, .4)
    im1 = ImageChops.blend(im3, im1, .4)

    return im1


def graph_hist(rememberhisto):
    (rememberhistoR, rememberhistoG, rememberhistoB) = rememberhisto
    (line, ) = plt.plot(np.array(rememberhistoR), 'r-', linewidth=2)
    (line, ) = plt.plot(np.array(rememberhistoG), 'g-', linewidth=2)
    (line, ) = plt.plot(np.array(rememberhistoB), 'b-', linewidth=2)
    plt.show()


def imgtouxmRGB(tomimg):
    (r, g, b) = tomimg.split()
    r = r.resize((4098, 4098))
    g = g.resize((4098, 4098))
    b = b.resize((4098, 4098))
    trdataR = r.getdata()
    trdataG = g.getdata()
    trdataB = b.getdata()
    uxmR = np.array(trdataR).reshape(4098, 4098)
    uxmG = np.array(trdataG).reshape(4098, 4098)
    uxmB = np.array(trdataB).reshape(4098, 4098)
    del tomimg

    return (uxmR, uxmG, uxmB)


def imguxret(gray):
    im = Image.fromarray(np.uint8(gray))

    return im


def fib(i, a, b):
    retfib = [a, b]
    while b < i:
        (a, b) = (b, a + b)
        retfib.append(b)


# Load an image
xView_dir = '../data/xView/'
train_dir = xView_dir + 'train_images/'
chip_name = '104.tif'
size_tuple = (4098, 4098)
testimg = Image.open(train_dir + chip_name)
testimg = testimg.resize(size=size_tuple)


mimgR = Image.new('L', testimg.size)
mimgG = Image.new('L', testimg.size)
mimgB = Image.new('L', testimg.size)


import os

png_dir = xView_dir + 'png/'
os.makedirs(name=png_dir, exist_ok=True)
file_suffix = '_' + chip_name.split('.')[0] + '.png'
for i in [2, 3, 5, 8, 13]:
    img = feqencyextractor_rot(testimg, i, 3, 257 - i, 4)
    img2 = feqencyextractor_rot_mirror(testimg, i, 3, 257 - i, 4)
    im3R = ImageChops.lighter(img2[0], img[0])
    im3G = ImageChops.lighter(img2[1], img[1])
    im3B = ImageChops.lighter(img2[2], img[2])
    mimgR = ImageChops.lighter(im3R, mimgR)
    mimgG = ImageChops.lighter(im3G, mimgG)
    mimgB = ImageChops.lighter(im3B, mimgB)
    
    finimg = Image.merge('RGB', (mimgR, mimgG, mimgB))
    
    file_name = 'phasemap' + str(i) + file_suffix
    print('Saving {}'.format(file_name))
    finimg.save(png_dir + file_name)
    
    finimg = ImageChops.multiply(finimg, testimg)
    file_name = 'phasemulti' + str(i) + file_suffix
    print('Saving {}'.format(file_name))
    finimg.save(png_dir + file_name)
    
    finimg = ImageChops.blend(finimg, testimg, .5)
    finimg = ImageChops.blend(finimg, testimg, .5)
    file_name = 'phaseblend' + str(i) + file_suffix
    print('Saving {}'.format(file_name))
    finimg.save(png_dir + file_name)
    
    testimg = finimg



