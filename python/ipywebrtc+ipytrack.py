import ipywebrtc as webrtc
import ipyvolume as ipv
import ipywidgets as widgets

video = webrtc.VideoStream(filename='big-buck-bunny_trailer.webm')

video

camera = webrtc.CameraStream()

camera

fig = ipv.figure(render_continuous=True)
back = ipv.plot_plane("back", texture=video)
right = ipv.plot_plane("right", texture=camera)
ipv.show()

right.texture = fig

room1 = webrtc.chat(room='demo', stream=fig)

room2 = webrtc.chat(room='demo', stream=camera)

room1.close()
room2.close()
#fig.render_continuous = False
camera.close()

video_track = webrtc.VideoStream(filename='head.webm')

video_track

import ipytrack as track

headtracker = track.HeadTrackr(stream=video_track)

headtracker

fig

widgets.jslink((headtracker, 'head'), (fig, 'camera_center'))



headtracker.scale = 0.1

recorder = webrtc.MediaRecorder(stream=fig, filename='record')

recorder.record = True

recorder.record = False

with open('sample.webm', 'wb') as f:
    f.write(recorder.data)

get_ipython().system('open sample.webm')



