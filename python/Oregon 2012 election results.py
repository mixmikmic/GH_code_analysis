import agate

results = agate.Table.from_csv('20121106__or__general.csv')

print results

offices = results.where(lambda r: r['candidate'] != 'Under Votes' and r['candidate'] != 'Over Votes').group_by('office')

office_and_party = offices.group_by('party')

op_totals = office_and_party.aggregate([('vote_total', agate.Sum('votes'))])

op_totals = op_totals.order_by('vote_total', reverse=True)

op_totals.print_table()

office_party_county = office_and_party.group_by('county')

office_party_county_totals = office_party_county.aggregate([('vote_total', agate.Sum('votes'))])

office_party_county_totals = office_party_county_totals.order_by('vote_total', reverse=True)

office_party_county_totals.print_table()

democratic_office_totals = office_party_county_totals.where(lambda r: r['party'] == 'D')

president_by_county = democratic_office_totals.where(lambda r: r['office'] == 'President')

print president_by_county.column_names

others_by_county = democratic_office_totals.where(lambda r: r['office'] != 'President')

more_than_prez = democratic_office_totals.where(lambda r: r['vote_total'] > [x['vote_total'] for x in president_by_county.rows 
                                                                           if r['county'] == x['county']][0])

more_than_prez.print_table(max_rows=50)

