import io
import csv
import datapackage
import jsontableschema as tableschema  # name has recently been updated for v1.
import goodtables

descriptor = 'data/israel-muni/datapackage.json'

dp = datapackage.DataPackage(descriptor)

# The loaded Descriptor
dp.descriptor

# The loaded Data Resource objects
dp.resources

# Each resource provides a stream over the data
israel_muni_budget_tree = dp.resources[0].iter()

israel_muni_budget_tree

# When a Data Resource is a Tabular Data Resource
# Values from the CSV are cast on iteration
tel_aviv_budget = dp.resources[1].iter()

get_ipython().run_cell_magic('bash', '', 'head -2 data/israel-muni/tel-aviv-2013.csv')

next(tel_aviv_budget)

get_ipython().run_cell_magic('bash', '', 'head -2 data/mailing-list/data.csv')

source = 'data/mailing-list/data.csv'
# When we just have a source of data, we can still get a schema
with io.open(source) as stream:
    reader = csv.reader(stream)
    headers = next(reader)
    values = list(reader)

schema = tableschema.infer(headers, values)

schema

# we can validate any schema
tableschema.validate(schema)

# and catch if a schema is not valid
try:
    tableschema.validate({"fields": {}})
except tableschema.exceptions.SchemaValidationError as e:
    msg = e.message

msg

# We get get some helper methods to work we schemas
model = tableschema.Schema(schema)

model.headers

model.has_field('occupation')

model.cast_row(['Amos', 'Levy', '13', '2.0', 'T', '2011-02-05'])

# We can iterate over a stream and cast values
table = tableschema.Table(source, schema)

next(table.iter())

next(table.iter(keyed=True))

# We saw the basics of handling Data Packages in the first demo.
# Now let's use our infered schema, and source data for a new Tabular Data Package

tdp = datapackage.DataPackage(schema='tabular')
tdp.descriptor

# We've just got an empty descriptor, so it is not actually valid
try:
    tdp.validate()
except datapackage.exceptions.ValidationError as e:
    msg = e.message

msg

# Add the minimum for a Tabular Data Resource
tdp.descriptor.update({
    'name': 'my-mailing-lists',
    'resources': [
        {
            'name': 'mailer',
            'path': source,
            'schema': schema
        }
    ]
})

tdp.validate()
tdp.descriptor

next(tdp.resources[0].iter())

inspector = goodtables.Inspector()
inspector.inspect('data/invalid.csv')

# We can customize our inspector
inspector = goodtables.Inspector(checks={
    'duplicate-header': False,
    'extra-header': False,
    'missing-value': False,
    'blank-header': False
})
inspector.inspect('data/invalid.csv')

# We can also inspect all Data Resources in a Data Package
inspector = goodtables.Inspector()
result = inspector.inspect('data/israel-muni/datapackage.json', preset='datapackage')

result['error-count'], result['table-count']

get_ipython().run_cell_magic('bash', '', 'goodtables')

