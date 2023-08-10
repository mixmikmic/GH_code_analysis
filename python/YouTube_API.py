import pafy
url = "https://www.youtube.com/watch?v=BGBM5vWiBLo"
video = pafy.new(url)

video.title

video.description

details = [video.title, video.rating, video.viewcount, video.author, video.length]
print(details)

# downloading the video with best quality
best_video = video.getbest()
best_video.download(quiet=False)

# downloading the audio of the video with best quality
bestaudio = video.getbestaudio()
bestaudio.download()

# getting all streams: all possible audio/video extensions
allstreams = video.allstreams
from pprint import pprint
pprint(allstreams)

for i in allstreams:
    print(i.mediatype, i.extension, i.quality)

# download a chosen filetype, e.g. m4a
allstreams[-3].download()

