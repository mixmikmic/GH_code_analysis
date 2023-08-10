import requests
import xml.etree.ElementTree as xml
from IPython.display import HTML, display
from Bio import Entrez

#email Set the Entrez email parameter (default is not set).
Entrez.email = "great_team@hackathon.ncbi.org"

#tool Set the Entrez tool parameter (default is biopython).
Entrez.tool = "hackathon_examples"

def get_nuccore_id(uid):
    """
    Get nuccore id by its refseq id.
    """
    url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi'
    params = {'dbfrom': 'assembly', 'db':'nuccore', 'retmode':'json', 'id': uid}
    r = requests.get(url, params=params)
    if r.status_code != 200:
        raise Exception("Cant nuccore id for specified assembly")
    doc = r.json()['linksets'][0]['linksetdbs']
    for link in doc:
        if link['linkname'] == "assembly_nuccore_refseq":
            return int(link['links'][0])
    else:
        return int(doc[0]['links'][0])

    
def list_gene_ids(nuc_id):
    """
    List genes for specified organism nuccore id.
    """
    url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi'
    params = {'dbfrom':'nuccore', 'db': 'gene', 'retmode': 'json', 'id':nuc_id}
    r = requests.get(url, params=params)
    if r.status_code != 200:
        raise Exception("Cant find annotation")
    return r.json()['linksets'][0]['linksetdbs'][0]['links']


def get_asm_details(accession):
    """
    Get ref_seq id and organism name for specified assembly.
    """
    params = {'release':accession}
    r = requests.get(
        'https://www.ncbi.nlm.nih.gov/projects/r_gencoll/asm4portal/gc_assembly_svc.cgi',
        params=params)
    if r.status_code != 200:
        raise Exception("Cant find assembly")
    doc = r.content.decode('utf-8')
    root = xml.fromstring(doc)
    refseq_id = int(root.attrib['uid'])
    header = root.find('header')
    organism_name = header.find('species-organism').text
    return refseq_id, organism_name


def list_genes(accession, max_show = 100):
    """
    For the given organism assembly build a table with first few genes.
    """
    refseq_id, organism_name = get_asm_details(accession)

    html = "<h1>" + organism_name + "</h1>"
    
    handle = Entrez.esearch(db="pubmed",term="'" + organism_name + "'")
    search_results =Entrez.read(handle)

    html += "<p>" + str(search_results['Count']) + " pubmeds found</p>" 

    nuccore_id = get_nuccore_id(refseq_id)

    genes = list_gene_ids(nuccore_id)

    html += "<h3> Genes: </h3>"

    html += "<table>"

    html += "<tr>"
        
    html += "<th> Gene Tag </th>"
    html += "<th> Gene Locus </th>"
    html += "<th> Protein </th>"
    html += "<th> Protein sequence </th>"
    html += "<th> Number of Pubmed publications </th>"
    html += "</tr>"
    for gene in genes[0:max_show]:
        handle = Entrez.efetch(db="gene", id=gene, retmode="xml")
        root = xml.fromstring(handle.read())
        gene_node = root.find('Entrezgene')
        gene_ref = gene_node.find('Entrezgene_gene').find('Gene-ref')
        
        locus_at = gene_ref.find('Gene-ref_locus')
        locus_tag = gene_ref.find('Gene-ref_locus-tag')

        terms = []
        locus = "n/a"
        tag = "n/a"

        try:
            tag = locus_tag.text
            terms.append(tag)
        except:
            pass
        
        try:
            locus = locus_at.text
            terms.append(locus)
        except:
            pass
        
        prot_node = gene_node.find('Entrezgene_prot')
        
        prot = "n/a"
        
        if prot_node:
            prot = prot_node.find('Prot-ref').find('Prot-ref_name').find('Prot-ref_name_E').text
            terms.append(prot)
            
            
        loc_node = gene_node.find('Entrezgene_locus')
        com = loc_node.find('Gene-commentary')
        prod = com.find('Gene-commentary_products')
        prod_seq = "n/a"
        prod_ver = "n/a"
        if prod:
            prod_com = prod.find('Gene-commentary')
            if prod_com:
                prod_seq = prod_com.find('Gene-commentary_accession').text
                prod_ver = prod_com.find('Gene-commentary_version').text
                terms.append(prod_seq)
            
        # Lookup pubs
        query = " or ".join(["'%s'" % (i) for i in terms])
        handle = Entrez.esearch(db="pubmed",term="(" + query + ") and '"+ organism_name + "'")
        search_results =Entrez.read(handle)

        html += "<tr>"
        
        html += "<td>" + tag + "</td>"
        html += "<td>" + locus + "</td>"
        html += "<td>" + prot + "</td>"
        html += "<td>" + prod_seq + "." + prod_ver + "</td>"
        html += "<td>" + str(search_results['Count']) + "</td>" 

        html += "</tr>"
    html += "</table>"

    if len(genes) > max_show:
        html += "<p>And " + str(len(genes) - max_show) + " more... </p>"

    display(HTML(html))

get_ipython().run_cell_magic('time', '', "\nlist_genes('GCF_000013425.1', 10)")



