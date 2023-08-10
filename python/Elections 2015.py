import json, pandas, requests
from datetime import datetime
import numpy
from pymldb import Connection
mldb = Connection()

ds = mldb.v1.datasets("raw")
ds.put({
    "type": "text.csv.tabular",
    "params": {
        "dataFileUrl":"file:///mldb_data/press_releases.csv.gz",
        "ignoreBadLines": True,
        "named": "'pr' + lineNumber()",
    }
})

mldb.query("""
SELECT party, count(*) 
FROM raw 
WHERE to_timestamp(date) > to_timestamp('2015-08-01')
GROUP BY party
""")

mldb.query("SELECT * FROM raw LIMIT 1")

bag = mldb.v1.procedures("baggify").put({
    "type": "transform",
    "params": {
        "inputDataset": "raw",
        "outputDataset": {
            "id": "bag_of_words",
            "type": "sparse.mutable"
        },
        "select": """
            tokenize(full_text, 
                        {splitchars: ' ?!;/[]*"',
                         quotechar: ''}
                    ) as *
        """,
        "where": """full_text IS NOT NULL AND 
                    title IS NOT NULL AND 
                    to_timestamp(date) > to_timestamp('2015-08-01')
        """,
        "runOnCreation": True
    }
})

mldb.query("SELECT * FROM bag_of_words LIMIT 5")

df = mldb.query("SELECT sum({*}) as * FROM bag_of_words")
df2 = df.T
df2.columns = ["count"]

df2.sort(columns="count", ascending=False)[:15]

w2v = mldb.v1.procedures("w2vimport").put({
    "type": 'import.word2vec',
    "params": {
        "dataFileUrl": 'file:///mldb_data/GoogleNews-vectors-negative300.bin',
        "outputDataset": {
            "type": 'embedding',
            "id": 'w2v'
        },
        "runOnCreation": True
    }
})

mldb.query("SELECT * FROM w2v LIMIT 5")

w2v = mldb.v1.functions("pooler").put({
    "type": "pooling",
    "params": {
        "aggregators": ["avg"],
        "embeddingDataset": "w2v"
    }
})

print mldb.v1.procedures("word2vec").put({
    "type": "transform",
    "params": {
        "inputDataset": "bag_of_words",
        "outputDataset": {
            "id": "pr_word2vec",
            "type": "sparse.mutable"
        },
        "select": "pooler({words: {*}}) as word2vec",
        "runOnCreation": True
    }
})

mldb.query("SELECT * FROM pr_word2vec LIMIT 5")

print mldb.v1.procedures("pr_embed_tsne").put({
    "type" : "tsne.train",
    "params" : {
        "trainingData" : "SELECT * FROM pr_word2vec",
        "rowOutputDataset" : "pr_embed_tsne",
        "modelFileUrl": "file:///mldb_data/tsne.bin",
        "functionName": "tsne_embed",
        "perplexity": 5,
        "runOnCreation": True
    }
})

mldb.v1.datasets("pr_embed_tsne_merged").put({
    "type" : "merged",
    "params" : {
        "datasets": [
            {"id": "raw"},
            {"id": "pr_embed_tsne"}
        ]
    }
})

mldb.query("""
SELECT party, title, x, y 
FROM pr_embed_tsne_merged 
WHERE to_timestamp(date) > to_timestamp('2015-08-01') 
LIMIT 5""")

df = mldb.query("""
SELECT party, title, x, y
FROM pr_embed_tsne_merged
WHERE to_timestamp(date) > to_timestamp('2015-08-01')
""")

import numpy as np
colormap = {
    "ndp": "#FF8000",
    "liberal": "#DF0101",
    "conservative": "#0000FF",
    "green": "#01DF01",
    "category": "#FE2EC8"
}

import bokeh.plotting as bp
from bokeh.models import HoverTool

#this line must be in its own cell 
bp.output_notebook()

press_releases = np.array([str(x.encode('ascii','ignore').split("|")[0]) for x in list(df.title.values)])
x = bp.figure(plot_width=900, plot_height=700, title="Press Releases of Canadian Federal Parties During 2015 Elections",
       tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
       x_axis_type=None, y_axis_type=None, min_border=1)
x.scatter(
    x = df.x.values, 
    y = df.y.values, 
    color=[colormap[k] for k in df.party.values],
    radius=1,
    fill_alpha=0.5,
    source=bp.ColumnDataSource({"title": press_releases})
).select(dict(type=HoverTool)).tooltips = {"title":"@title"}
bp.show(x)

