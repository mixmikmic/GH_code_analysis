import os
import sys
os.chdir('../')
sys.path.insert(0,os.getcwd())

from graphviz import Digraph, Source
from IPython.display import display
def show_dot(dot_file):
    display(Source(open(dot_file).read()), unconfined=False)

show_dot("examples_cbnets/alarm.dot")

show_dot("examples_cbnets/asia.dot")

show_dot("examples_cbnets/barley.dot")

show_dot("examples_cbnets/cancer.dot")

show_dot("examples_cbnets/child.dot")

show_dot("examples_cbnets/earthquake.dot")

show_dot("examples_cbnets/hailfinder.dot")

show_dot("examples_cbnets/HuaDar.dot")

show_dot("examples_cbnets/insurance.dot")

show_dot("examples_cbnets/mildew.dot")

show_dot("examples_cbnets/Monty_Hall.dot")

show_dot("examples_cbnets/sachs.dot")

show_dot("examples_cbnets/student.dot")

show_dot("examples_cbnets/survey.dot")

show_dot("examples_cbnets/water.dot")

show_dot("examples_cbnets/WetGrass.dot")



