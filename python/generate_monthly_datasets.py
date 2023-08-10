get_ipython().system('pip install git+https://github.com/ipython/ipynb.git')
get_ipython().system('pip install pymysql')

from ipynb.fs.full.article_quality.db_monthly_stats import DBMonthlyStats, dump_aggregation

import configparser
config = configparser.ConfigParser()
config.read('../settings.cfg')

import os
def write_once(path, write_to):
    if not os.path.exists(path):
        print("Writing out " + path)
        with open(path, "w") as f:
            write_to(f)
    

dbms = DBMonthlyStats(config)

write_once(
    "../data/processed/enwiki.full_wiki_aggregation.tsv", 
    lambda f: dump_aggregation(dbms.all_wiki_aggregation(), f))

write_once(
    "../data/processed/enwiki.wikiproject_women_scientists_aggregation.tsv", 
    lambda f: dump_aggregation(dbms.wikiproject_aggregation("WikiProject_Women_scientists"), f))

write_once(
    "../data/processed/enwiki.wikiproject_oregon_aggregation.tsv", 
    lambda f: dump_aggregation(dbms.wikiproject_aggregation("WikiProject_Oregon"), f))



