import brightway2 as bw
import wurst as w
import pandas as pd
import csv

db = bw.Database("ecoinvent 3.3 cutoff")

df = pd.read_excel("stem-to-ecoinvent.xlsx")
df

labels = [x[0].strip() for x in df.values]

mapped = {
    label.strip(): w.get_one(db, 
                             w.equals('name', name.strip()), 
                             w.equals('location', location.strip()),
                             w.equals('unit', 'kilowatt hour'))
    for label, name, location in df.values
}
mapped

selected_methods = [
    ('IPCC 2013', 'climate change', 'GWP 100a'),
    ('ReCiPe Endpoint (H,A)', 'total', 'total'),
]

results = []

for method in selected_methods:
    for technology in mapped:
        lca = bw.LCA({mapped[technology]: 1}, method)
        lca.lci()
        lca.lcia()
        results.append(("-".join(method), technology, lca.score))

with open('stem-scores.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    for row in results:
        writer.writerow(row)

