import agate

ies = agate.Table.from_csv('in.csv')

print ies

purposes = ies.group_by('purpose')

purpose_totals = purposes.aggregate([('purpose_total', agate.Sum('amount'))])

purpose_totals = purpose_totals.order_by('purpose_total', reverse=True)

purpose_totals.print_table(max_column_width=50, max_rows=100)

spenders = ies.group_by('fec_committee_id')

spender_totals = spenders.aggregate([('spender_total', agate.Sum('amount'))])

spender_totals = spender_totals.order_by('spender_total', reverse=True)

spender_totals.print_table(max_column_width=50, max_rows=25)



