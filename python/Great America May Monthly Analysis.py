import agate

contributions = agate.Table.from_csv('great_america_sa.csv')

print contributions

print contributions.aggregate(agate.Max('amount'))

top_donor = contributions.where(lambda row: row['amount'] == 50000.0)

print top_donor.rows[0]['last_name'] +', '+ top_donor.rows[0]['first_name'] + ' - ' + top_donor.rows[0]['city'] + ', ' + top_donor.rows[0]['state'] 

print contributions.aggregate(agate.Count('transaction_id'))

print contributions.aggregate(agate.Median('amount'))

no_emp_occ = contributions.where(lambda row: row['employer'] == 'INFORMATION REQUESTED PER BEST EFFORTS')

print no_emp_occ.aggregate(agate.Count('transaction_id'))

print (90.0/125.0)*100



