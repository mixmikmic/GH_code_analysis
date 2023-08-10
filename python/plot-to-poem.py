import random
import sys
import textwrap

import plotutils
import poemutils

import spacy
from annoy import AnnoyIndex
import numpy as np

nlp = spacy.load('en')

titles = plotutils.titleindex()
plots = plotutils.loadplots()
assert(len(titles) == len(plots))

def getplot(idx):
    return idx, titles[idx], plots[idx]
def pickrandom(titles, plots):
    idx = random.randrange(len(titles))
    return getplot(random.randrange(len(titles)))

pickrandom(titles, plots)

title2idx = dict([(t, i) for i, t in enumerate(titles)])
title2idx["New Super Mario Bros."]

getplot(title2idx["New Super Mario Bros."])

def meanvector(text):
    s = nlp(text)
    vecs = [word.vector for word in s             if word.pos_ in ('NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN', 'ADP')             and np.any(word.vector)] # skip all-zero vectors
    if len(vecs) == 0:
        raise IndexError
    else:
        return np.array(vecs).mean(axis=0)
meanvector("this is a test").shape

t = AnnoyIndex(300, metric='angular')
lines = list()
i = 0
for line in poemutils.loadlines(modulo=20):
    if i % 10000 == 0:
        sys.stderr.write(str(i) + "\n")
    try:
        t.add_item(i, meanvector(line['line']))
        lines.append(line)
        i += 1
    except IndexError:
        continue

t.build(25)

assert(t.get_n_items() == len(lines))

nearest = t.get_nns_by_vector(meanvector("All that glitters is gold"), n=10)[0]
print(lines[nearest]['line'])

idx, title, sentences = pickrandom(titles, plots)
print(title)
print('-' * len(title))
print()
for sent in sentences.split("\n"):
    try:
        vec = meanvector(sent)
    except IndexError:
        continue
    match_idx = t.get_nns_by_vector(vec, n=100)[0]
    print(textwrap.fill(sent+".", 60))
    print("\t", lines[match_idx]['line'])
    print()

import html

titles_to_try = [
    "Star Wars (film)",
    "When Harry Met Sally...",
    "House of Leaves",
    "Shrek",
    "The Hobbit",
    "The Legend of Zelda: Ocarina of Time",
    "The Handmaid's Tale",
    "Ferris Bueller's Day Off",
    "Star Trek II: The Wrath of Khan",
    "Lost in Translation (film)",
    "The Matrix",
    "Doom (1993 video game)",
    "Neuromancer",
    "Top Gun",
    "A Wrinkle in Time",
    "The Wizard of Oz (1939 film)"
]

html_tmpl = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Plot to Poem: Sample output</title>
    <style type="text/css">
        .line {{
            cursor: pointer;
            font-family: serif;
            font-size: 12pt;
            line-height: 1.5em;
        }}
        .line:hover {{
            background-color: #f8f8f8;
        }}
        body {{
            margin: 2em auto;
            width: 67%;
            font-family: sans-serif;
        }}
        h2 {{
            margin-top: 2em;
            margin-bottom: 0.5em;
        }}
    </style>
</head>

<body>

<h1>Plot to poem</h1>
<p>By <a href="http://www.decontextualize.com/">Allison Parrish</a>
    for <a href="https://github.com/NaPoGenMo/NaPoGenMo2017">NaPoGenMo 2017</a>.</p>
<p>Each sentence from the Wikipedia plot summary of these
    works has been replaced with the line of poetry from
    Project Gutenberg that is closest in meaning. Mouse over
    the line to see the original sentence from the plot
    summary.</p>
<p>Want to learn more about how it works, or try it out on
    your own text?
    <a href="https://github.com/aparrish/plot-to-poem/">Python
    source code available here.</a></p>

{poems}

</body>
</html>
"""

output_html = ""
for to_try in titles_to_try:
    already_seen = set()
    idx, title, sentences = getplot(title2idx[to_try])
    output_html += "\n<h2>"+title+"</h2>"
    for sent in sentences.split("\n"):
        try:
            vec = meanvector(sent)
        except IndexError:
            continue
        match_idx = [i for i in t.get_nns_by_vector(vec, n=100) if i not in already_seen][0]
        already_seen.add(match_idx)
        output_html += "<div title='{orig}' class='line'>{line}</div>\n".format(
            orig=html.escape(sent), line=lines[match_idx]['line'])
open("plot-to-poem.html", "w").write(html_tmpl.format(poems=output_html))

