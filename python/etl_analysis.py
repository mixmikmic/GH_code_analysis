import os
import subprocess
import shutil
import sys

ROOT_FOLDER = os.getcwd()

# no optimizations
get_ipython().run_line_magic('timeit', "-n1 -r3 subprocess.call('pytest tests/unit/test_models.py', shell=True)")

# with optimizations
import os
from tests.conftest import KEEPDB_PATH

# ensure the first run creates the retained folder
testdb_path = os.path.join(ROOT_FOLDER, KEEPDB_PATH)
if os.path.exists(testdb_path):
    shutil.rmtree(testdb_path)

get_ipython().run_line_magic('timeit', "-n1 -r3 subprocess.call('pytest --keepdb tests/unit/test_models.py', shell=True)")

