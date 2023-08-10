import os
import shutil
import json
from pprint import pprint
from glob import iglob

# --------------------------------------------------------
# This cell defines various utility functions
# --------------------------------------------------------
def ignore_dir(d):
    """Returns whether the input string contains one of the ignore pattern"""
    #TODO: The input arg d must be a string! Add exception handling code. 
    ignore_patterns = ['dsx', 'images']
    for p in ignore_patterns:
        if p.lower() in d.lower():
            return True
    return False

def ignore_file(f):
    """Returns whether the input string contains one of the ignore pattern"""
    ignore_patterns = ['index.ipynb', 'dsx']
    for p in ignore_patterns:
        if p.lower() in f.lower():
            return True
    return False

def copy_notebook_tree(d_names, d_fullpath):
    """Recursively make copies the Quantum notebooks to be customized for IBM DSX. 
    
    Copies the whole directory tree of all the Quantum notebooks. The copy of the
    directory tree is placed inside the folder, qiskit-tutorial/ibm_dsx
    """
    for dname, src_dir in zip(d_names, d_fullpath):
        print("dname=",dname)
        
        try:
            # First recursively remove the existing directories in this folder
            shutil.rmtree(os.path.abspath(dname))
        except FileNotFoundError as e:
            print("ignoring, dir {} does not exist".format(dname))
        except:
            print("unknown error")
            raise
        #Copy the whole directory tree to "./" (i.e. qiskit-tutorial/ibm_dsx)
        shutil.copytree(src_dir, os.path.abspath(dname))
        
def create_dsx_patch():
    """Prepare the 'patch' (the code cell) required to run the notebook on IBM DSX.
    
    It is extracted from from qikit-tutorial/1_introduction/running_on_IBM_DSX.ipynb
    Returns the json string representing the patch to be inserted in other notebooks. 
    
    :return: dsx_patch 
    """
    with open("../1_introduction/running_on_IBM_DSX.ipynb") as fil:
        src_data = json.load(fil)

    n = len(src_data['cells'])

    dsx_patch = None

    for i in range(n):
        if src_data['cells'][i]['cell_type'] == "code":
            dsx_patch = src_data['cells'][i]
            #print("index to insert the patch is:", i)
            break

    assert dsx_patch is not None
    
    return dsx_patch


def customize_for_dsx(fname, dsx_patch):
    """The workhorse method that patches the given notebook with the DSX specific customization.
    
    :arg fname: File path string (the Jupyter notebook to be modified)
    :arg dsx_patch: The json formatted dsx patch to be inserted as the 'first code' cell 
           into the input file. 
           
    In the end, it just overwrites the Jupyter notebook file. 
    
    .. todo: Nice to have some error handling code.
    """
    # ---------------------------------------------------------------------------------------
    # We will be modify destination_data later by inserting the customization specific to 
    # IBM Data Science Experience. 
    # ---------------------------------------------------------------------------------------
    with open(fname) as fil:
        destination_data = json.load(fil)
    
    n = len(destination_data['cells'])
    idx = None
    
    for i in range(n):
        if destination_data['cells'][i]['cell_type'] == "code":
            idx = i
            break
            
    print("index to insert the dsx patch is:", idx)
    
    # Now insert the patch into the original notebook (dst_data)
    destination_data['cells'].insert(idx, dsx_patch)
    
    # Overwrite the file
    with open(fname, "w") as fil:
        fil.write(json.dumps(destination_data))

# Time to execute the code.

# The input directories are one level up relative to this script (if not , modify the following code!)
# This script is supposed to be here qiskit-tutorial/ibm_dsx/apply_dsx_patch.ipynb
# One level up is: qiskit-tutorial/
# ---------------------------------------------------------------------------------------
d_fullpath = [d for d in iglob('../**', recursive=False) if os.path.isdir(d) and not ignore_dir(d) ]
d_names = [os.path.basename(d) for d in d_fullpath]
print(d_fullpath)
print(d_names)

# Copy the notebooks
copy_notebook_tree(d_names, d_fullpath)

# Prepare the 'dsx patch' 
dsx_patch = create_dsx_patch()

# Do the customization
for dname in d_names:
    # List notebooks with their directory path 
    path_str = '%s/*.ipynb'%(dname)
    initial_notebooks = [f for f in iglob(path_str, recursive=True) if os.path.isfile(f)]
    print("-"*50)
    print("dname: ", dname)
    final_notebooks= [f for f in initial_notebooks if not ignore_file(f)]
    print(final_notebooks)
    for f in final_notebooks:
        customize_for_dsx(f, dsx_patch)
        

