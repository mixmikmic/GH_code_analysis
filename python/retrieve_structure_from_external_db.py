from aiida import load_dbenv, is_dbenv_loaded
if not is_dbenv_loaded():
    load_dbenv()
from aiida.tools.dbimporters import DbImporterFactory
from ase.visualize import view
import spglib
import nglview

importer_class = DbImporterFactory('cod')
importer = importer_class()
importer

cod_id = '1510230' # AuMg
# cod_id = '9009138' # CdI2
# cod_id = '9008845' # GaAs

query_results = importer.query(formula='Au Mg')

print len(query_results)

entry = query_results.at(0)
cif = entry.get_cif_node()
print cif.get_formulae()

structure = cif._get_aiida_structure(converter='pymatgen')

structure.store()
structure.pk

print structure.get_formula()
print structure.get_ase()

view = nglview.show_ase(structure.get_ase()*[4,4,1])
view

print spglib.get_spacegroup(structure.get_ase())

get_ipython().system('ssh -fN -L 3306:localhost:3306 -L 8010:localhost:80 aiidademo@theossrv2.epfl.ch > /dev/null 2>&1')

importer_class = DbImporterFactory('icsd')
importer_parameters = {'server': 'http://localhost:8010/',
                   'host': '127.0.0.1',
                   'db': 'icsd',
                   'passwd': 'sql'
                  }
importer = importer_class(**importer_parameters)
importer

icsd_id = '617290' # graphite

query_results = importer.query(id=icsd_id)

print len(query_results)

entry = query_results.at(0)
cif = entry.get_cif_node()
print cif.get_formulae()

structure = cif._get_aiida_structure(converter='pymatgen')
print structure.get_formula()
print structure.get_ase()

print spglib.get_spacegroup(structure.get_ase(),symprec=5e-3)

structure.store()
structure.pk

view = nglview.show_ase(structure.get_ase()*[4,4,1])
view





