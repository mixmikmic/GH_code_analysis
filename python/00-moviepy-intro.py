from moviepy.editor import *
from IPython.display import HTML

MAX_LENGTH = 1
screensize = (500,281)

shrug = VideoFileClip("res/shrug.gif").subclip(0, MAX_LENGTH)

caption = TextClip('WHY NOT?',
                    color='white', font="Impact", kerning = -3, 
                    fontsize=75, stroke_color='black', stroke_width=3) 

video = CompositeVideoClip( [shrug, caption.set_pos('bottom').set_duration(MAX_LENGTH)] )

video.write_videofile("shrug.mp4")

HTML("""
<video width="500" height="500" style="margin:auto auto" autoplay loop="true" >
  <source src="shrug.mp4" type="video/mp4">
</video>
""")



