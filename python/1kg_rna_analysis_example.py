import scipy.stats
from ga4gh.client import client
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
local_client = client.HttpClient("http://1kgenomes.ga4gh.org")
#local_client = client.HttpClient("http://localhost:8000")

dataset_id = local_client.search_datasets().next().id
rna_quantification_sets = []
for rna_quant_set in local_client.search_rna_quantification_sets(dataset_id=dataset_id):
    rna_quantification_sets.append(rna_quant_set.id)
    print("id: {}".format(rna_quant_set.id))
    print("name: {}\n".format(rna_quant_set.name))

rna_set_id = rna_quantification_sets[0]
rna_quantification_ids = []
counter = 0
max_num_quantifications = 5
for rna_quant in local_client.search_rna_quantifications(rna_quantification_set_id=rna_set_id):
    if counter < max_num_quantifications:
        # In order to make later examples run faster we will store the IDs of the first
        # few quantifications returned by the server.
        rna_quantification_ids.append(rna_quant.id)
    counter += 1
    print("({}): {}\n".format(rna_quant.id, rna_quant.name))

feature_sets = set()
for rna_quant_id in rna_quantification_ids:
    for feature_set_id in local_client.get_rna_quantification(
            rna_quantification_id=rna_quant_id).feature_set_ids:
        feature_sets.add(feature_set_id)
print("If == 1 we don't have to cull from the list --> {}".format(len(feature_sets)))

def getUnits(unitType):
    units = ["", "FPKM", "TPM"]
    return units[unitType]

counter = 0
expression_levels = []
for expression in local_client.search_expression_levels(
        rna_quantification_id=rna_quantification_ids[0]):
    if counter > 5:
        break
    counter += 1
    if expression.feature_id != "":
        expression_levels.append(expression)
    print("Expression Level: {}".format(expression.name))
    print(" id: {}".format(expression.id))
    print(" feature: {}".format(expression.feature_id))
    print(" expression: {} {}".format(expression.expression, getUnits(expression.units)))
    print(" read_count: {}".format(expression.raw_read_count))
    print(" confidence_interval: {} - {}\n".format(
            expression.conf_interval_low, expression.conf_interval_high))

feature_ids = [expression_levels[1].feature_id]
for rna_quantification_id in rna_quantification_ids[1:]:
    for expression in local_client.search_expression_levels(
            rna_quantification_id=rna_quantification_id, feature_ids=feature_ids):
        print("RNA Quantification: {}".format(rna_quantification_id))
        print("Expression Level: {}".format(expression.name))
        print(" id: {}".format(expression.id))
        print(" feature: {}".format(expression.feature_id))
        print(" expression: {} {}\n".format(expression.expression, getUnits(expression.units)))

def build_expression_dict(rna_quantification_id, max_features=50):
    counter = 0
    expression_dict = {}
    for expression in local_client.search_expression_levels(
            rna_quantification_id=rna_quantification_id):
        if counter > max_features:
            break
        counter += 1
        if expression.feature_id != "":
            expression_dict[expression.name] = expression.expression
    return expression_dict


expressions_dict_1 = build_expression_dict(rna_quantification_ids[0])
featureNames = set(expressions_dict_1.keys())
expressions_dict_2 = build_expression_dict(rna_quantification_ids[1])
featureNames = featureNames.intersection(set(expressions_dict_2.keys()))
sample_1 = []
sample_2 = []
featureNameList = list(featureNames) # preserve feature order
print("Comparing {} features".format(len(featureNameList)))
for feature_name in featureNameList:
    sample_1.append(expressions_dict_1[feature_name])
    sample_2.append(expressions_dict_2[feature_name])

scipy.stats.spearmanr(sample_1, sample_2)

utils = rpackages.importr('utils')
utils.chooseBioCmirror(ind=1)
robjects.r.source("https://bioconductor.org/biocLite.R")
robjects.r.biocLite(robjects.r.c("RColorBrewer", "pheatmap", "DESeq2"))

def build_expression_dict(rna_quantification_id, max_features=50):
    """
        We are going to rewrite this to return count as well as the quantification
        name so that we can build the required matrix.  Also, zero count features
        are going to be filtered out.
    """
    counter = 0
    expression_dict = {}
    quantification = local_client.get_rna_quantification(
        rna_quantification_id=rna_quantification_id)
    for expression in local_client.search_expression_levels(
            rna_quantification_id=rna_quantification_id):
        if counter > max_features:
            break
        counter += 1
        if expression.feature_id != "" and expression.raw_read_count > 0:
            expression_dict[expression.name] = expression.raw_read_count
    return quantification.name, expression_dict


rna_quant_names = []
rna_quant_values = []
conditions = []
for i in range(5):
    quant_name, expressions_dict = build_expression_dict(rna_quantification_ids[i])
    rna_quant_names.append(quant_name)
    rna_quant_values.append(expressions_dict)
    conditions.append("sample_{}".format(i+1))
    if i == 0:
        featureNames = set(expressions_dict.keys())
    else:
        featureNames = featureNames.intersection(set(expressions_dict.keys()))

def get_count_vector(featureList, quantification):
    """
        Extracts counts from the quantification dictionary using featureList as keys.
        Returns an robjects.IntVector of counts.
    """
    count_list = [quantification[feature_name] for feature_name in featureList]
    return robjects.IntVector(count_list)


featureNameList = list(featureNames) # preserve feature order
countData = robjects.r.cbind(get_count_vector(featureNameList, rna_quant_values[0]),
                             get_count_vector(featureNameList, rna_quant_values[1]))
for quantification in rna_quant_values[2:]:
    countData = robjects.r.cbind(countData, get_count_vector(featureNameList, quantification))
countData.rownames = robjects.StrVector(featureNameList)
countData.colnames = robjects.StrVector(rna_quant_names)
coldata = robjects.r.cbind(robjects.StrVector(rna_quant_names),
                           robjects.StrVector(conditions))
coldata.colnames = robjects.StrVector(["library", "sample"])
print("matrix created")

robjects.r.library("DESeq2")
design = robjects.Formula("~ library")
ddsMat = robjects.r.DESeqDataSetFromMatrix(countData=countData, colData=coldata,
                                           design=design)
deseq_result = robjects.r.DESeq(ddsMat)
# robjects.r.results shows the last result
print(robjects.r.results(deseq_result))
# can also look any of the results by specifying contrast:
print(robjects.r.results(deseq_result, contrast=robjects.r.c("library", rna_quant_names[0],
                                                             rna_quant_names[1])))

robjects.r.X11()
robjects.r.library("pheatmap")
rld = robjects.r.rlog(deseq_result, blind=False)
print(robjects.r.head(robjects.r.assay(rld), 3))
robjects.r.pheatmap(robjects.r.assay(rld))



