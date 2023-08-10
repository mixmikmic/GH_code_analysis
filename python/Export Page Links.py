import json
import os
from collections import defaultdict
from epub_conversion import convert_wiki_to_lines
import gzip
from epub_conversion.wiki_decoder import almost_smart_open
from multiprocessing import Process, Lock
from multiprocessing import Pool

import re
link_pattern = re.compile(r'\[\[ *(.*?)\]\]')

def save_progress(work_so_far, path, num_saved, mode):
    with open(path, mode) as fout:
        for ex in work_so_far:
            json.dump(ex, fout)
            fout.write("\n")
    num_saved = num_saved + len(work_so_far)
    return num_saved
        
def lines_extractor(lines, article_name):
    yield (article_name, lines)
    
def category_job(args):
    article_name, lines = args
    out = []
    text_block = True
    for block in re.split(link_pattern, lines):
        block = block.strip()
        if text_block:
            text_block = False
            out.append({"type":"text", "text": block})
        else:
            link = block
            if '|' in link:
                link, anchor = link.split("|", 1)
                link = link.strip().split("#")[0]
                anchor = anchor.strip()
                
                if link.startswith("File:") or link.startswith("Image:"):
                    if len(anchor) > 0:
                        out.append({"type":"text", "text": anchor})
                elif len(link) > 0 and len(anchor) > 0:
                    anchor_words = anchor.split(" ")
                    out.append({"type":"label", "text": anchor, "label": link})
                elif len(anchor) > 0:
                    out.append({"type":"text", "text": anchor})
            else:
                if len(link) > 0:
                    out.append({"type":"label", "text": link, "label": link})
            text_block = True
    return (article_name, out)

def run_jobs(worker_pool, pool_jobs, output):
    results = worker_pool.map(category_job, pool_jobs)
    for article_name, out in results:
        output.append(
            {
                "content": out,
                "title": article_name
            }
        )

def parse_wiki(path, outpath, num_articles=9999999999999, threads=1, max_jobs=10, save_every=10000):
    num_articles_processed       = 0
    num_articles_with_categories = 0
    processed_categories = []
    
    jobs = []
    pool = Pool(processes=threads)
    try:
        num_saved = 0
        write_mode = "wt+"

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

                if num_articles_processed % save_every == 0:
                    num_saved = save_progress(processed_categories, outpath, num_saved, mode=write_mode)
                    processed_categories = []
                    write_mode = "at+"

        if len(jobs) > 0:
            run_jobs(pool, jobs, processed_categories)
            jobs = []
        num_saved = save_progress(processed_categories, outpath, num_saved, mode=write_mode)
        processed_categories = []
        write_mode = "at+"
    finally:
        pool.close()
        pool.join()
    return processed_categories

x = parse_wiki(
    "/Users/jonathanraiman/Desktop/Coding/enwiki2015.xml.bz2",
    "/Users/jonathanraiman/Desktop/datasets/triggers_and_documents.json",
    threads=9,
    max_jobs=100,
    save_every=100000
)



