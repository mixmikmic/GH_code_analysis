import os
import sys
os.chdir('../')
sys.path.insert(0,os.getcwd())

from graphviz import Digraph, Source

Source(open("examples_cbnets/alarm.dot").read())

Source(open("examples_cbnets/asia.dot").read())

Source(open("examples_cbnets/barley.dot").read())

Source(open("examples_cbnets/cancer.dot").read())

Source(open("examples_cbnets/child.dot").read())

Source(open("examples_cbnets/earthquake.dot").read())

Source(open("examples_cbnets/hailfinder.dot").read())

Source(open("examples_cbnets/HuaDar.dot").read())

Source(open("examples_cbnets/insurance.dot").read())

Source(open("examples_cbnets/mildew.dot").read())

Source(open("examples_cbnets/Monty_Hall.dot").read())

Source(open("examples_cbnets/sachs.dot").read())

Source(open("examples_cbnets/student.dot").read())

Source(open("examples_cbnets/survey.dot").read())

Source(open("examples_cbnets/water.dot").read())

Source(open("examples_cbnets/WetGrass.dot").read())



