import wurst as w
import brightway2 as bw

data = w.extract_brightway2_databases(["ecoinvent 3.3 cutoff"])

electricity_unit = [w.equals("unit", "kilowatt hour")]
natural_gas = electricity_unit + [w.contains("name", "natural gas")]
combined_cycle = natural_gas + [w.contains("name", "combined cycle")]

combined_cycle(data)

w.get_many(data, *combined_cycle)

len(list(w.get_many(data, *combined_cycle)))

len(list(w.get_many(data, w.equals("unit", "kilowatt hour"), w.contains("name", "natural gas"))))

for ds in w.get_many(data, *combined_cycle):
    w.change_exchanges_by_constant_factor(ds, 0.9, biosphere_filters=[w.contains('name', 'water')])

ng_in_shangdong = w.get_one(data, w.equals('location', 'CN-SD'), *combined_cycle)
our_exc = w.get_one(w.biosphere(ng_in_shangdong), w.equals('name', 'Carbon dioxide, fossil'))
w.rescale_exchange(our_exc, 1. / our_exc['amount'])

ds = {
    'location': ('ecoinvent', 'UN-NEUROPE'),
    'exchanges': [{
        'name': 'A', 'product': 'B', 'unit': 'C',
        'amount': 10,
        'type': 'technosphere',
    }]
}
given_data = [{
    'name': 'A', 'reference product': 'B', 'unit': 'C',
    'location': 'SE',
    'exchanges': [{
        'type': 'production', 'amount': 1,
        'production volume': 2,
    }]
}, {
    'name': 'A', 'reference product': 'B', 'unit': 'C',
    'location': 'NO',
    'exchanges': [{
        'type': 'production', 'amount': 1,
        'production volume': 4,
    }]
}]
new_data = [{
    'name': 'A', 'reference product': 'B', 'unit': 'C',
    'location': 'RoW', # RoW means allocation by production volume isn't possible, instead split evenly
    'exchanges': [{
        'type': 'production', 'amount': 1,
        'production volume': 14,
    }]
}, {
    'name': 'D', 'reference product': 'E', 'unit': 'F',
    'location': 'DK', # Right location but wrong activity
    'exchanges': [{
        'type': 'production', 'amount': 1,
        'production volume': 1,
    }]
}]
w.relink_technosphere_exchanges(ds, given_data + new_data)



