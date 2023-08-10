import pandas as pd
import numpy as np
from glob import glob
from collections import defaultdict, Counter

from urlparse import urlsplit, parse_qs

import re

classification_files = glob("DomainDataset/*+suffix.txt")
print classification_files

CAT_REGEX = re.compile(r'.*/([a-zA-Z]+)_.*')

url_categories = defaultdict(set)
for filename in classification_files:
    catname = CAT_REGEX.match(filename).groups()[0].lower()
    if catname == "fakenewschecker":
        catname = "fakenews"
    print "%s\t%s" % (filename, catname)
    with open(filename) as fp:
        for line in fp:
            line = line.strip().lower()
            if line.startswith("www."):
                line = line[4:]
            url_categories[line].add(catname)
            
len(url_categories), url_categories["facebook.com"]
url_categories["twitter.com"].add("twitter") # Manually add twitter in seperate category

wikidata_files = glob("DomainDataset/Wikidata_*.tsv")
print wikidata_files

WIKIDATA_CAT_REGEX = re.compile(r'.*/.*_([a-zA-Z\ ]+).*')

for filename in wikidata_files:
    catname = WIKIDATA_CAT_REGEX.match(filename).groups()[0].lower()
    print "%s\t%s" % (filename, catname)
    with open(filename) as fp:
        header = fp.readline()
        for line in fp:
            line = line[:-1].lower().split("\t")[-1]
            if line.strip() == "":
                continue
            try:
                line = line.split("/", 3)[2]
            except:
                print line
                raise
            if line.startswith("www."):
                line = line[4:]
            url_categories[line].add(catname)

CAT_MAPPINGS={
    "satire": "fakenews",
    "clickbait": "fakenews",
    "usgov": "news"
}
pd.Series(
    Counter(
        sum((list(CAT_MAPPINGS.get(x, x) for x in k)
             for k in url_categories.itervalues()),
            []))).to_frame().reset_index().rename(
    columns={0: "Counts",
            "index": "URL category"})

df_t = pd.Series(url_categories)
df_t[(df_t.apply(lambda k: len(set(CAT_MAPPINGS.get(x, x) for x in k))) > 1)]

with open("DomainDataset/URL_CATS.txt", "wb+") as fp:
    for url, cats in url_categories.iteritems():
        print >> fp, "%s\t%s" % (url, ",".join(cats))
        
get_ipython().system(' head DomainDataset/URL_CATS.txt')

df_url_counts = pd.read_csv("all_urls.txt", sep="\t", header=None)
df_url_counts.columns = ["URL", "DOMAIN", "Counts"]
df_url_counts.head()

df = pd.read_csv("url_expanded.merged.txt", sep="\t")
df.head()

"http://linkis.com/freebeacon.com/polit/3Fjdv".split("/", 1)

parse_qs(urlsplit("https://www.google.com/url?rct=j&sa=t&url=http://www.phoenixnewtimes.com/news/harkins-theaters-cancel-arizona-showing-of-anti-vaccine-film-8255215&ct=ga&cd=CAIyGjE2ZDBhYmZjOTAzMjkyMTk6Y29tOmVuOlVT&usg=AFQjCNHJWqaVm8jBMMQhMe39xm5Wtiy-3A").query)

def get_url_domain(x):
    x = urlsplit(x.lower())
    if x.netloc in {"linkis.com", "www.linkis.com"}:
        if x.path[1:] != "":
            x = urlsplit("http:/%s" % x.path).netloc
        else:
            x = x.netloc
    elif x.netloc in {"google.com", "www.google.com"}:
        query = parse_qs(x.query)
        if "url" in query:
            return get_url_domain(query["url"][0])
        x = x.netloc
    else:
        x = x.netloc
    if x.startswith("www."):
        x = x[4:]
    if x.endswith(".wordpress.com") or x.endswith(".tumblr.com") or x.endswith(".blogspot.com"):
        x = x.split(".", 1)[-1]
    return x

get_url_domain("https://www.google.com/url?rct=j&sa=t&url=http://www.perthnow.com.au/news/western-australia/social-services-minister-christian-porter-slaps-down-antivaccination-campaigners/news-story/0aa49052ec0598704b05333075581296&ct=ga&cd=CAIyGjE2ZDBhYmZjOTAzMjkyMTk6Y29tOmVuOlVT&usg=AFQjCNFAB3aZtdfdVpXOHWzyfqsu0ZSFAg")

df["URL_DOMAIN"] = df.EXPANDED.apply(get_url_domain)
df.head()

df["URL_CATS"] = df.URL_DOMAIN.apply(lambda x: url_categories.get(x, "UNK"))
df.head()

df[df.URL_CATS != "UNK"].head()

df[df.URL_CATS != "UNK"].shape, df.shape

df[df.URL_CATS == "UNK"].head(10)

df[df.URL_DOMAIN == "com"].head()

df[df.URL_CATS == "UNK"].URL_DOMAIN.value_counts()

df_url_counts = df_url_counts.merge(df, how="inner", on="URL")
df_url_counts.shape

df_url_counts.head()

df_url_counts[df_url_counts.URL_CATS == "UNK"].groupby("URL_DOMAIN")["Counts"].first().sort_values(ascending=False).head(10)

df.assign(
    URL_CATS = lambda x: x.URL_CATS.apply(lambda cats: "|".join(cats) if cats != "UNK" else cats)
).to_csv("URL_CAT_MAPPINGS.txt", sep="\t", index=False)
get_ipython().system(' head URL_CAT_MAPPINGS.txt')

reduce(lambda x, y: x.union(y), url_categories.values())

df.shape

df[df.URL_DOMAIN == 'paper.li'].EXPANDED.head().values



