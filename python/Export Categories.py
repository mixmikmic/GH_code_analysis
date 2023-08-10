import json
from collections import defaultdict
from epub_conversion import convert_wiki_to_lines
import gzip
from epub_conversion.wiki_decoder import almost_smart_open
from multiprocessing import Process, Lock
from multiprocessing import Pool

import re
link_pattern = re.compile(r'\[\[ *(.*?)\]\]')

def save_progress(work_so_far, path):
    with open(path, "w+") as fout:
        json.dump(work_so_far, fout)
        
def lines_extractor(lines, article_name):
    yield (article_name, lines)
    
def category_job(args):
    article_name, lines = args
    categories = []
    for link in link_pattern.findall(lines):
        if link.lower().startswith("category:"):
            if '|' in link:
                link, anchor = link.split("|", 1)
                link = link.strip().split("#")[0]
                anchor = anchor.strip()
                if len(link) > 0 and len(anchor) > 0:
                    categories.append((link, anchor))
            else:
                categories.append((link, None))
    return (article_name, categories)

def run_jobs(worker_pool, pool_jobs, output_dictionary):
    results = worker_pool.map(category_job, pool_jobs)
    for article_name, categories in results:
        for category, anchor in categories:
            output_dictionary[category].append(article_name)

def parse_wiki(path, outpath, num_articles = 9999999999999, threads = 1, max_jobs = 10):
    num_articles_processed       = 0
    num_articles_with_categories = 0
    processed_categories = defaultdict(lambda : [])
    
    jobs = []
    pool = Pool(processes=threads)
    
    with almost_smart_open(path, "rb") as wiki:
        for article_name, lines in convert_wiki_to_lines(
                wiki,
                max_articles         = num_articles,
                clear_output         = True,
                report_every         = 100,
                parse_special_pages  = True,
                skip_templated_lines = False,
                line_converter       = lines_extractor):
            
            jobs.append((article_name, lines))
            
            num_articles_processed += 1
            
            if len(jobs) >= max_jobs:
                run_jobs(pool, jobs, processed_categories)
                jobs = []

            if num_articles_processed % 100000 == 0:
                save_progress(processed_categories, outpath)
    
    if len(jobs) > 0:
        run_jobs(pool, jobs, processed_categories)
        jobs = []
    
    save_progress(processed_categories, outpath)
    
    return processed_categories

x = parse_wiki("/Users/jonathanraiman/Desktop/Coding/enwiki2015.xml.bz2",
           "/Users/jonathanraiman/Desktop/datasets/category_graph2.json",
           threads=9,
           max_jobs=100)



