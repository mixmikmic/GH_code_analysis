# we need to install suds and nvd3 to display our results
get_ipython().system('pip install --user --quiet suds')
get_ipython().system('pip install --user --quiet python-nvd3')

# set your registered email address here
email = ''

import sys
import pandas
from StringIO import StringIO
from suds.client import Client
david_wsdl_url = 'http://david.abcc.ncifcrf.gov/webservice/services/DAVIDWebService?wsdl'
client = Client(david_wsdl_url)
registered = client.service.authenticate(email)

# set your input data here; For example in Galaxy you could enter `get(4)`
uniprot = pandas.read_csv('/home/bag/Downloads/uniprot-cytochrome.tab', sep='\t')

get_ipython().run_cell_magic('javascript', '', 'require.config({paths: {d3: "//d3js.org/d3.v3.min"}});')

from IPython.display import HTML
from nvd3 import pieChart
import nvd3
nvd3.ipynb.initialize_javascript(use_remote=True)

# define a plotting fucntion based on d3.js and nvd3
def pie_chart(x, y, name='piechart'):
    """
    x and y are lists of values and label
    name needs to be different between different plots, otherwise one plot overwrites the other
    """
    chart = pieChart(name=name, color_category='category20c', height=650, width=650)
    chart.set_containerheader("\n\n<h2>PieChart</h2>\n\n")
    xdata = x
    ydata = y
    extra_serie = {"tooltip": {"y_start": "", "y_end": " score"}}
    chart.add_serie(y=ydata, x=xdata, extra=extra_serie)
    chart.buildcontent()
    return chart.htmlcontent 

def david_setup(input_ids, id_type='UNIPROT_ACCESSION', 
                bg_ids=[], bg_name='IPython_bg_name',
                list_name='IPython_example_list', category=''):
    """
    possible categories:
        * BBID,GOTERM_CC_FAT,BIOCARTA,GOTERM_MF_FAT,SMART,COG_ONTOLOGY,SP_PIR_KEYWORDS,
        KEGG_PATHWAY,INTERPRO,UP_SEQ_FEATURE,OMIM_DISEASE,GOTERM_BP_FAT,PIR_SUPERFAMILY
    
    """
    david = client.service
    input_ids = ','.join(input_ids)
    if bg_ids:
        bg_ids = ','.join(bg_ids)

    list_type = 0
    print 'Percentage mapped: %s' % david.addList(input_ids, id_type, list_name, list_type)
    if bg_ids:
        list_type = 1
        print 'Percentage mapped (background): %s' % david.addList(bg_ids, id_type, bg_name, list_type)

    david.setCategories(category)
    return david

def report_to_table(request):
    """
    Converts a DAVID report to a pandas DataFrame.
    """
    results = list()
    for row in request:
        results.append(dict(row))
    df = pandas.DataFrame()
    return df.from_dict(results)

david = david_setup(uniprot['Entry'][:100], 'UNIPROT_ACCESSION', category='GOTERM_CC_FAT')

ct = 2
thd = 0.1
request = david.getChartReport(thd, ct)
table = report_to_table(request)
table[['categoryName','termName', 'listHits', 'percent', 'ease', 'foldEnrichment', 'benjamini']]

overlap = 2
initialSeed = 2
finalSeed = 1
linkage = 1
kappa = 1
request = david.getGeneClusterReport(overlap, initialSeed, finalSeed, linkage, kappa)
table = report_to_table(request)
table[['name', 'score']]

overlap = 3
initialSeed = 3
finalSeed = 3
linkage = 0.5
kappa = 50
request = david.getTermClusterReport(overlap, initialSeed, finalSeed, linkage, kappa)
table = report_to_table(request)
table[['name', 'score']]

HTML(pie_chart(table['name'], table['score'], name="relaxed"))

overlap = 5
initialSeed = 5
finalSeed = 5
linkage = 0.5
kappa = 50 
request = david.getTermClusterReport(overlap, initialSeed, finalSeed, linkage, kappa)
table = report_to_table(request)
print table[['name', 'score']]

HTML(pie_chart(table['name'], table['score']))

