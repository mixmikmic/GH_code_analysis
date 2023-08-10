get_ipython().magic('reload_ext autoreload')
get_ipython().magic('autoreload 2')
def warn(*args, **kwargs):
    pass  # to suppress sklearn warnings

import warnings
warnings.filterwarnings("ignore")
warnings.warn = warn

from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole

# The next two lines are for optical reasons only. They can be safely disabled.
Draw.DrawingOptions.atomLabelFontFace = "DejaVu Sans"
Draw.DrawingOptions.atomLabelFontSize = 18

from rdkit_ipynb_tools import tools, pipeline as p, sar

get_ipython().system('zcat chembl_et-a_antagonists.txt.gz | wc -l')
print()
get_ipython().system('zcat chembl_et-a_antagonists.txt.gz | head -n 1')

s = p.Summary()  # optional, used for logging what the individual components do

# code for IC50 --> pIC50 conversion
run_code = """rec["ETA_pIC50"] = tools.pic50(rec["STANDARD_VALUE"], "nM")"""  

# define the start of the pipeline, can work directly with gzipped files
rd = p.start_csv_reader("chembl_et-a_antagonists.txt.gz", summary=s)

et_a_list = p.pipe(rd,
             (p.pipe_has_prop_filter, "STANDARD_VALUE", {"summary": s}),
             (p.pipe_custom_man, run_code),
             (p.pipe_keep_props, ["CMPD_CHEMBLID", "CANONICAL_SMILES", "ETA_pIC50"]),
             (p.pipe_rename_prop, "CMPD_CHEMBLID", "Chembl_Id"),
             (p.pipe_mol_from_smiles, "CANONICAL_SMILES"),
             (p.pipe_calc_props, ["2d", "LogP", "MW"]),
             (p.stop_mol_list_from_stream, {"max": 3000, "summary": s})
            )
s.update(True)

mol_list = et_a_list.prop_filter("MW <= 500").prop_filter("MW > 300")

mol_list.summary()

for mol in mol_list:
    if float(mol.GetProp("ETA_pIC50")) >= 7.0:
        mol.SetProp("AC_Real", "1")
    else:
        mol.SetProp("AC_Real", "0")

train_list, test_list = mol_list.split(0.25)
print(len(train_list), len(test_list))

model = sar.train(train_list)

test_list = sar.SAR_List(test_list)
test_list.order = ["Chembl_Id", "ETA_pIC50", "AC_Real", "AC_Pred", "Prob", "LogP"]

test_list.model = model

test_list.predict()

test_list.summary()

show_props = ["ETA_pIC50", "AC_Real", "AC_Pred", "Prob"]
test_list.grid(props=show_props)

_ = test_list.analyze()

sample_list = test_list.sample(10)
sample_list.remove_props("MW")
sample_list.sort_list("ETA_pIC50")
sample_list.sim_map()

sample_list.write_sim_map()



