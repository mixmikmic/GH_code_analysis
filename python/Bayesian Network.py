from pgmpy.factors import TabularCPD
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination

A_cpd = TabularCPD(
    variable='A',
    variable_card=2,
    values=[[0.2, 0.8]])

B_cpd = TabularCPD(
    variable='B',
    variable_card=2,
    values=[[0.9, 0.1]])

C_cpd = TabularCPD(
    variable='C',
    variable_card=2,
    values=[[1-0.05, 1-0.60, 1-0.75, 1-0.90],
            [1-0.95, 1-0.40, 1-0.25, 1-0.10]],
    evidence=['B', 'A'],
    evidence_card=[2, 2])

D_cpd = TabularCPD(
    variable='D',
    variable_card=2,
    evidence = ['B'],
    values = [[0.25, 0.80],
             [0.75, 0.20]], 
    evidence_card = [2])

E_cpd = TabularCPD(
    variable='E',
    variable_card=2,
    evidence = ['C'],
    values = [[0.95, 0.15],
             [0.05, 0.85]], 
    evidence_card = [2])

gene_model = BayesianModel([('A', 'C'),
                            ('B', 'C'),
                            ('C', 'E'),
                            ('B', 'D')])
gene_model.add_cpds(A_cpd,B_cpd,C_cpd,D_cpd,E_cpd)

gene_infer = VariableElimination(gene_model)
prob_C = gene_infer.query(variables='C')

print(prob_C['C'])



