import sys
import os
import getpass
from IPython.display import display, HTML
import ipywidgets as widgets
sys.path.insert(0, '../code')
import mcl_ui_utils as ui_utils

from odm2api.ODMconnection import dbconnection
from odm2api.ODM2.models import *

#print("Enter your ODM2 username") 
container = ui_utils.ODM2LoginPrompt()
container

print("enter your password: ")
p = getpass.getpass()

#createConnection(self, engine, address, db=None, user=None, password=None, dbtype = 2.0, echo=False)
session_factory = dbconnection.createConnection('postgresql', container.children[1].value, 
                                                container.children[2].value, 
                                                container.children[0].value, p)   
DBSession = session_factory.getSession()

variable = Variables(VariableTypeCV='Chemistry', VariableCode = 'Tl-particulate', VariableNameCV='Thalium, particulate',
                    VariableDefinition='particulate thallium quantified by ICP MS', SpeciationCV= 'Tl', NoDataValue=-6999)
print(variable)
print(variable.VariableCode)
DBSession.add(variable)
DBSession.commit()
print("the ID value for our new variable")
print(variable.VariableID)
variable_id = variable.VariableID

retreived_variable = DBSession.query(Variables).get(variable_id)
print(retreived_variable)
DBSession.delete(retreived_variable)
DBSession.commit()



