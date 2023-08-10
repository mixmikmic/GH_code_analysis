import pandas as pd
import requests
from wikidataintegrator import wdi_core
pd.set_option('display.max_colwidth', -1)

query = """SELECT ?hgnc ?protein ?go ?goLabel ?goId
WHERE
{
  values ?hgnc {"FANCA" "FANCB" "FANCC" "FANCE" "FANCF" "FANCG" "FANCL" "FANCM" "FANCD2" "FANCI" "UBE2T" "FANCD1" "BRCA2" "FANCJ" "FANCN" "FANCO" "FANCP" "FANCQ" "FANCR" "FANCS" "FANCV" "FANCU" "FAAP100" "FAAP24" "FAAP20" "FAAP16" "MHF1" "FAAP10" "MHF2"}
  ?gene wdt:P353 ?hgnc .  # get gene items with these HGNC symbols
  ?gene wdt:P688 ?protein . # get the protein
  ?protein wdt:P680|wdt:P681|wdt:P682 ?go . # get GO terms
  ?go wdt:P686 ?goId
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
}"""
d = wdi_core.WDItemEngine.execute_sparql_query(query)

df = pd.DataFrame([{k:v['value'] for k,v in x.items()} for x in d['results']['bindings']])

df

def f(x):
     return pd.Series(dict(goLabel = list(x['goLabel'])[0], 
                        hgnc = ','.join(x['hgnc']),
                          count = len(x)))

df2 = df.groupby("goId").apply(f).sort_values("count", ascending=False)

df2

