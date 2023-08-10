import sys
import logging
import time
import getopt
import json
import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')

#load in dataset
df = pd.read_csv("../data/movie_ratings_simple.csv")

nltk.download() # Download the "stopwords" corpus

# Build a random title
titlelist = " ".join(df['title'])
import re
# Use regular expressions to do a find-and-replace
letters_only = re.sub("[^a-zA-Z]",           # The pattern to search for
                      " ",                   # The pattern to replace it with
                      titlelist )  # The text to search

words = [x.lower().strip() for x in letters_only.split(" ")]
words = [x for x in words if x != '']
import nltk

from nltk.corpus import stopwords # Import the stop word list
#print(stopwords.words("english") )
stops = set(stopwords.words("english")) 
# Remove stop words
meaningful_words = [w for w in words if not w in stops]   
def getTitle(meaningful_words):
    wordseries=pd.Series(meaningful_words)
    newstring = ' '.join(wordseries.sample(n=np.random.randint(1,5)))
    prefixrand = np.random.randint(0,4)
    if prefixrand == 2:
        newstring = 'the ' + newstring
    elif prefixrand == 3:
        newstring = 'a ' + newstring
    return newstring.title()

def getGenres(df):
    #Get genre options:
    genrechoices = []
    for k in range(4):
        g1 = df['Genre1'].sample(n=1).values[0]
        g2 = g1
        g3 = g1
        while g2 == g1:
            g2 = df['Genre2'].sample(n=1).values[0]
        if g2 == 'None':
            g3 = 'None'
        else:
            g3 = g1
            g3 = g2
            while g3 == g1 or g3 == g2:
                g3 = df['Genre3'].sample(n=1).values[0]
        genrechoices.append([g1,g2,g3])
    return genrechoices

def on_draw(event):
    """Auto-wraps all text objects in a figure at draw-time"""
    import matplotlib as mpl
    fig = event.canvas.figure

    # Cycle through all artists in all the axes in the figure
    for ax in fig.axes:
        for artist in ax.get_children():
            # If it's a text artist, wrap it...
            if isinstance(artist, mpl.text.Text):
                autowrap_text(artist, event.renderer)

    # Temporarily disconnect any callbacks to the draw event...
    # (To avoid recursion)
    func_handles = fig.canvas.callbacks.callbacks[event.name]
    fig.canvas.callbacks.callbacks[event.name] = {}
    # Re-draw the figure..
    fig.canvas.draw()
    # Reset the draw event callbacks
    fig.canvas.callbacks.callbacks[event.name] = func_handles

def autowrap_text(textobj, renderer):
    """Wraps the given matplotlib text object so that it exceed the boundaries
    of the axis it is plotted in."""
    import textwrap
    # Get the starting position of the text in pixels...
    x0, y0 = textobj.get_transform().transform(textobj.get_position())
    # Get the extents of the current axis in pixels...
    clip = textobj.get_axes().get_window_extent()
    # Set the text to rotate about the left edge (doesn't make sense otherwise)
    textobj.set_rotation_mode('anchor')

    # Get the amount of space in the direction of rotation to the left and 
    # right of x0, y0 (left and right are relative to the rotation, as well)
    rotation = textobj.get_rotation()
    right_space = min_dist_inside((x0, y0), rotation, clip)
    left_space = min_dist_inside((x0, y0), rotation - 180, clip)

    # Use either the left or right distance depending on the horiz alignment.
    alignment = textobj.get_horizontalalignment()
    if alignment is 'left':
        new_width = right_space 
    elif alignment is 'right':
        new_width = left_space
    else:
        new_width = 2 * min(left_space, right_space)

    # Estimate the width of the new size in characters...
    aspect_ratio = 0.5 # This varies with the font!! 
    fontsize = textobj.get_size()
    pixels_per_char = aspect_ratio * renderer.points_to_pixels(fontsize)

    # If wrap_width is < 1, just make it 1 character
    wrap_width = max(1, new_width // pixels_per_char)
    try:
        wrapped_text = textwrap.fill(textobj.get_text(), wrap_width)
    except TypeError:
        # This appears to be a single word
        wrapped_text = textobj.get_text()
    textobj.set_text(wrapped_text)

def min_dist_inside(point, rotation, box):
    """Gets the space in a given direction from "point" to the boundaries of
    "box" (where box is an object with x0, y0, x1, & y1 attributes, point is a
    tuple of x,y, and rotation is the angle in degrees)"""
    from math import sin, cos, radians
    x0, y0 = point
    rotation = radians(rotation)
    distances = []
    threshold = 0.0001 
    if cos(rotation) > threshold: 
        # Intersects the right axis
        distances.append((box.x1 - x0) / cos(rotation))
    if cos(rotation) < -threshold: 
        # Intersects the left axis
        distances.append((box.x0 - x0) / cos(rotation))
    if sin(rotation) > threshold: 
        # Intersects the top axis
        distances.append((box.y1 - y0) / sin(rotation))
    if sin(rotation) < -threshold: 
        # Intersects the bottom axis
        distances.append((box.y0 - y0) / sin(rotation))
    return min(distances)

#Ok, now we build the array of 36 choices as a new dataframe that we can then plot/work with

# First, get three directors (from the list of directors that have at least 3 films)
vc=df['Director1'].value_counts()
subdir = vc[vc>3]
directors = subdir.sample(n=3).index.values

# Now get three years
years = df['year'].sample(n=3).values

# Generate the random genres
genrechoices = getGenres(df)

datachoices = pd.DataFrame(columns=['title','Director1','year','Genre1','Genre2','Genre3'])
index = 0
for director in directors:
    for year in years:
        for choice in genrechoices:
            newdata = dict()
            newdata['title'] = getTitle(meaningful_words) + ' ({})'.format(year)
            newdata['Director1'] = [director]
            newdata['year'] = [str(year)]
            newdata['Genre1'] = [choice[0]]
            newdata['Genre2'] =[choice[1]]
            newdata['Genre3'] = [choice[2]]
            new_df = pd.DataFrame.from_dict(newdata)
            datachoices = datachoices.append(new_df, ignore_index=True)

datachoices.sort_values(['Director1','year','Genre1','Genre2','Genre3'],inplace=True)
datachoices.reset_index(drop=True,inplace=True)
sns.set_style("white")
import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(9, 4)
loc = 0
plt.figure(figsize=(10, 8))
for k in range(9):
    for j in range(4):
        ax = plt.subplot(gs[k, j])
        ax.text(0,0.8,datachoices['Director1'].ix[loc],ha='center', va='top')
        
        ax.text(0,0,datachoices['title'].ix[loc],horizontalalignment='center',ha='center', va='top')
        ax.text(0,-1.6,"/ ".join([x for x in datachoices[['Genre1','Genre2','Genre3']].ix[loc].values if x != 'None']),horizontalalignment='center',                 verticalalignment='top')

        ax.set_ylim([-3,2])
        ax.set_xlim([-3,3])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        loc += 1
        #plt.axis('off')
plt.gcf().canvas.mpl_connect('draw_event', on_draw)
plt.tight_layout(pad=0)

