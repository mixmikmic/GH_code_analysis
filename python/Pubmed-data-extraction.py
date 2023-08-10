from Bio import Entrez
import time
from urllib.error import HTTPError  # for Python 3

def search(query, num=2):
    Entrez.email = 'saifeng.mcmaster@gmail.com'
    Entrez.tool = 'WisOnePubMedProject'
    handle = Entrez.esearch(db='pubmed', 
                            sort='relevance', 
                            retmax=str(num),
                            retmode='xml',
                            usehistory='y',
                            term=query)
    results = Entrez.read(handle)
    handle.close()
    return results

# search_results=search('quantitative susceptibility mapping magnetic',300)
search_results=search('susceptibility+weighted+imaging or susceptibility+mapping ',1000)

count = int(search_results["Count"])
webenv = search_results["WebEnv"]
query_key = search_results["QueryKey"]
batch_size = 50
out_handle = open("SWI_QSM_papers.txt", "w")

for start in range(0, count, batch_size):
    end = min(count, start+batch_size)
    print("Downloading record %i to %i" % (start+1, end))
    attempt = 1
    while attempt <= 3:
        try:
            Entrez.email = 'saifeng.mcmaster@gmail.com'
            Entrez.tool = 'WisOnePubMedProject'
            fetch_handle = Entrez.efetch(db="pubmed", 
                                         rettype="medline", 
                                         retmode="text",
                                         retstart=start, 
                                         retmax=batch_size,
                                         webenv=webenv, 
                                         query_key=query_key)
            attempt = 4
            time.sleep(1)
        except HTTPError as err:
            if 500 <= err.code <= 599:
                print("Received error from server %s" % err)
                print("Attempt %i of 3" % attempt)
                attempt += 1
                time.sleep(15)
            else:
                raise
    data = fetch_handle.read()
    fetch_handle.close()
    out_handle.write(data)
out_handle.close()



