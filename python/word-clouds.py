from wordcloud import WordCloud, STOPWORDS
import matplotlib
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = (16.0, 9.0)

# Reading the script
script = open("wonderwoman.txt").read()
# Set of Stop words
stopwords = set(STOPWORDS)
stopwords.add("will")
# Create WordCloud Object
wc = WordCloud(background_color="white", stopwords=stopwords, 
               width=1600, height=900, colormap=matplotlib.cm.inferno)
# Generate WordCloud
wc.generate(script)
# Show the WordCloud
plt.figure()
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")

from PIL import Image
import numpy as np
from wordcloud import WordCloud, STOPWORDS
import matplotlib
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = (16.0, 9.0)

script = open("batman.txt").read()
stopwords = set(STOPWORDS)
batman_mask = np.array(Image.open("batman-logo.png"))

from matplotlib.colors import LinearSegmentedColormap
colors = ["#484848", "#000000", "#0060A8", "#FFF200", "#303030"]
cmap = LinearSegmentedColormap.from_list("mycmap", colors)

wc = WordCloud(background_color="white", stopwords=stopwords, mask=batman_mask,
               width=1987, height=736, colormap=cmap)
wc.generate(script)
plt.figure()
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = (16.0, 9.0)

mask = np.array(Image.open("civilwar.jpg"))
# Reading the script
script = open("civilwar.txt").read()
# Set of Stop words
stopwords = set(STOPWORDS)
# Create WordCloud Object
wc = WordCloud(background_color="white", stopwords=stopwords, 
               width=1280, height=628, mask=mask)
wc.generate(script)
# Image Color Generator
image_colors = ImageColorGenerator(mask)

plt.figure()
plt.imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")
plt.axis("off")

import random
from PIL import Image
import numpy as np
from wordcloud import WordCloud, STOPWORDS
import matplotlib
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = (16.0, 9.0)

# Custom Color Function
def grey_color_func(word, font_size, position, orientation, random_state=None,
                    **kwargs):
    return "hsl(0, 0%%, %d%%)" % random.randint(60, 100)

script = open("canon.txt").read()
stopwords = set(STOPWORDS)
stopwords.add("said")
stopwords.add("will")
mask = np.array(Image.open("sherlock.jpeg"))

wc = WordCloud(background_color="black", stopwords=stopwords, mask=mask,
               width=875, height=620,  font_path="lato.ttf")
wc.generate(script)
plt.figure()
plt.imshow(wc.recolor(color_func=grey_color_func, random_state=3), interpolation="bilinear")
plt.axis("off")

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = (16.0, 9.0)

mask = np.array(Image.open("trump.jpg"))
# Reading the script
script = open("trump.txt").read()
# Set of Stop words
stopwords = set(STOPWORDS)
stopwords.add("will")

from matplotlib.colors import LinearSegmentedColormap
colors = ["#BF0A30", "#002868"]
cmap = LinearSegmentedColormap.from_list("mycmap", colors)

# Create WordCloud Object
wc = WordCloud(background_color="white", stopwords=stopwords, font_path="titilium.ttf", 
               width=853, height=506, mask=mask, colormap=cmap)
wc.generate(script)


plt.figure()
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")

import random
from PIL import Image
import numpy as np
from wordcloud import WordCloud, STOPWORDS
import matplotlib
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = (16.0, 9.0)

# Custom Color Function
def grey_color_func(word, font_size, position, orientation, random_state=None,
                    **kwargs):
    return "hsl(0, 0%%, %d%%)" % random.randint(60, 100)

script = open("starwars.txt").read()
stopwords = set(STOPWORDS)
stopwords.add("will")
mask = np.array(Image.open("darthvader.jpg"))

wc = WordCloud(background_color="black", stopwords=stopwords, mask=mask,
               width=736, height=715,  font_path="lato.ttf")
wc.generate(script)
plt.figure()
plt.imshow(wc.recolor(color_func=grey_color_func, random_state=3), interpolation="bilinear")
plt.axis("off")

