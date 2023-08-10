from ga4gh.client import client
c = client.HttpClient("http://1kgenomes.ga4gh.org")

#Obtain dataSet id REF: -> `1kg_metadata_service`
dataset = c.search_datasets().next() 

counter = 0
for rna_quant_set in c.search_rna_quantification_sets(dataset_id=dataset.id):
    if counter > 5:
        break
    counter += 1
    print(" id: {}".format(rna_quant_set.id))
    print(" dataset_id: {}".format(rna_quant_set.dataset_id))
    print(" name: {}\n".format(rna_quant_set.name))

single_rna_quant_set = c.get_rna_quantification_set(
    rna_quantification_set_id=rna_quant_set.id)
print(" name: {}\n".format(single_rna_quant_set.name))

counter = 0
for rna_quant in c.search_rna_quantifications(
        rna_quantification_set_id=rna_quant_set.id):
    if counter > 5:
        break
    counter += 1
    print("RNA Quantification: {}".format(rna_quant.name))
    print(" id: {}".format(rna_quant.id))
    print(" description: {}\n".format(rna_quant.description))
    test_quant = rna_quant

single_rna_quant = c.get_rna_quantification(
    rna_quantification_id=test_quant.id)
print(" name: {}".format(single_rna_quant.name))
print(" read_ids: {}".format(single_rna_quant.read_group_ids))
print(" annotations: {}\n".format(single_rna_quant.feature_set_ids))

def getUnits(unitType):
    units = ["", "FPKM", "TPM"]
    return units[unitType]


counter = 0
for expression in c.search_expression_levels(
        rna_quantification_id=test_quant.id):
    if counter > 5:
        break
    counter += 1
    print("Expression Level: {}".format(expression.name))
    print(" id: {}".format(expression.id))
    print(" feature: {}".format(expression.feature_id))
    print(" expression: {} {}".format(expression.expression, getUnits(expression.units)))
    print(" read_count: {}".format(expression.raw_read_count))
    print(" confidence_interval: {} - {}\n".format(
            expression.conf_interval_low, expression.conf_interval_high))

counter = 0
for expression in c.search_expression_levels(
        rna_quantification_id=test_quant.id, feature_ids=[]):
    if counter > 5:
        break
    counter += 1
    print("Expression Level: {}".format(expression.name))
    print(" id: {}".format(expression.id))
    print(" feature: {}\n".format(expression.feature_id))

counter = 0
for expression in c.search_expression_levels(
        rna_quantification_id=test_quant.id, threshold=1000):
    if counter > 5:
        break
    counter += 1
    print("Expression Level: {}".format(expression.name))
    print(" id: {}".format(expression.id))
    print(" expression: {} {}\n".format(expression.expression, getUnits(expression.units)))



