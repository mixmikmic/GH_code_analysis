import py2neo
import pandas as pd

password = 'demo'
samples = 20

graph = py2neo.Graph(password)

query = """
    MATCH (d:Datasource)
    WHERE rand() <= 0.1
    RETURN Labels(d) as labels, d.name as name,
    d.about as about, d.link as link LIMIT {}
    """.format(samples)

df = pd.DataFrame(graph.data(query))
df = df.loc[:, ['name', 'about', 'link', 'labels']]
df

