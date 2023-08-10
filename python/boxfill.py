import vcs, cdms2
import os
if not os.path.exists(vcs.sample_data):
    vcs.download_sample_data_files()
f = cdms2.open(os.path.join(vcs.sample_data, "clt.nc"))
s = f('clt')
x = vcs.init(bg=True)
box = vcs.createboxfill()
x.plot(s, box)

box.list()

box.projection = "polar"
x.clear()
x.plot(s, box)

box.projection = "linear"
box.xticlabels1 = {0: "Prime Meridian", -121.7680: "Livermore", 37.6173: "Moscow"}
box.yticlabels1 = {0: "Eq.", 37.6819: "L", 55.7558: "M"}
x.clear()
x.plot(s, box)

box.xmtics1 = "lon30"  # Every 30deg
box.ymtics1 = "lat20"  # Every 20deg
template = vcs.createtemplate()
template.xmintic1.priority = 1
template.ymintic1.priority = 1
x.clear()
x.plot(s, box, template)

box.datawc_x1 = -125
box.datawc_x2 = -117
box.datawc_y1 = 34
box.datawc_y2 = 38
box.xmtics1 = box.xticlabels1
box.ymtics1 = box.yticlabels1
box.xticlabels1 = "*"
box.yticlabels1 = "*"
template.xmintic1.y1 = template.data.y1
template.xmintic1.y2 = template.data.y2
template.ymintic1.x1 = template.data.x1
template.ymintic1.x2 = template.data.x2
x.clear()
x.plot(s, box, template, ratio="autot")

box.datawc_x1 = 1e20
box.datawc_x2 = 1e20
box.datawc_y1 = 1e20
box.datawc_y2 = 1e20
box.xmtics1 = None
box.ymtics1 = None
box.level_1 = 20
box.level_2 = 40
x.clear()
x.plot(s, box, template)

box.color_1 = 60
box.color_2 = 80
x.clear()
x.plot(s, box, template)

box.level_1 = 1e20
box.level_2 = 1e20
box.color_1 = 0
box.color_2 = 255
box.boxfill_type = "custom"
x.clear()
x.plot(s, box, template)

box.levels = [10, 15, 20, 25, 30, 35]
box.fillareacolors = [0, 20, 40, 60, 80]
x.clear()
x.plot(s, box, template)

box.levels = [0, 25, 50, 75, 100]
box.fillareacolors = [0, 60, 120, 180]
box.fillareastyle = "hatch"  # or pattern
box.fillareaindices = [1, 2, 3, 4]
x.clear()
x.plot(s, box, template)

box.fillareaopacity = [50, 100, 100, 25]  # 0-100
box.fillareastyle = "pattern"
x.clear()
x.plot(s, box, template)

box.legend = {15: "This is a point", 30: "This is another point", 60: "One more"}
x.clear()
x.plot(s, box, template)

box.ext_1 = True
box.ext_2 = True
box.levels = [25, 50, 75]
box.fillareacolors = [100, 200]
box.fillareaindices = [1, 2]
box.fillareaopacity = [50, 100]
x.clear()
x.plot(s, box, template)

masked = cdms2.MV2.masked_less_equal(s, 30)
box.missing = "red"
box.fillareastyle = "solid"
box.levels = range(0, 101, 10)
box.fillareacolors = vcs.getcolors(box.levels)
box.fillareaindices = [1]
box.fillareaopacity = [100]
box.ext_1 = False
box.ext_2 = False
x.clear()
x.plot(masked, box, template)

