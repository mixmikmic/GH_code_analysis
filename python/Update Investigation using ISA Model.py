from isatools import isatab
from isatools.model import *

investigation = isatab.load(open('../data/BII-I-1/i_investigation.txt'), skip_load_tables=True)

print(isatab.dumps(investigation, skip_dump_tables=True))

investigation.identifier = "i1"
investigation.title = "My Simple ISA Investigation"
investigation.description = "We could alternatively use the class constructor's parameters to set some default "                             "values at the time of creation, however we want to demonstrate how to use the "                             "object's instance variables to set values."
investigation.submission_date = "2016-11-03"
investigation.public_release_date = "2016-11-03"

print(isatab.dumps(investigation, skip_dump_tables=True))



