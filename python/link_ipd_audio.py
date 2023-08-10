# Imports
import re
import os
import sys
import pardir; pardir.pardir() # Allow imports from parent directory

# User-defined paths

# HTML input and output
in_filepath = "../fibonaccistretch_with_figs.html"
out_filepath = "../index.html"
out_title = "Fibonacci Stretch"
# in_filepath = "../fibonaccistretch_examples.html"
# out_filepath = "../examples.html"
# out_title = "Fibonacci Stretch: Examples"

# Directory containing zero-indexed mp3s
data_dir = "../data/ipd_audio/"
# data_dir = "../data/ipd_audio_examples_nb/"
data_ext = "mp3"

# Directory containing figures
fig_path = "data/figs"

# Read input HTML
html = ""
with open(in_filepath, "r") as f:
    html = f.read()
len(html)

# Find <audio> elements
# p = re.compile('<audio controls="controls" >\s*<source src=".*" type=".*" \/>.*<\/audio>')
p = re.compile('<audio controls="controls" >\s*<source src=".*" type=".*" \/>')
matches = p.findall(html)
len(matches)

# Replace audio tag sources with corresponding mp3s
# (assumes that we have them all numbered from 0 to n-1 for n sources)
for i,m in enumerate(matches):
    mp3_filepath = os.path.join(data_dir, "{}.{}".format(i, data_ext))
    audio_tag = '<audio controls="controls"><source src="{}" type="audio/mp3" />'.format(mp3_filepath)
    # print(audio_tag)
    # print(m[:100])
    
    html = html.replace(m, audio_tag)
len(html)

# Replace fig path if necessary
# Unnecessary if using export_with_figs now
html = html.replace('img src="output', 'img src="{}/output'.format(fig_path))
len(html)

# Update title if necessary
html = re.sub("<title>.*</title>", "<title>{}</title>".format(out_title), html)
len(html)

# Update data path
html = html.replace("../data", "data")

with open(out_filepath, "w") as f:
    f.write(html)



