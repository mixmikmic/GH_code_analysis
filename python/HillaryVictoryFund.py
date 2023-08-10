import agate

skedb = agate.Table.from_csv('sb.csv')

print skedb

recipients = skedb.group_by('organization_name')

recipient_totals = recipients.aggregate([('recip_total', agate.Sum('amount'))])

recipient_total = recipient_totals.order_by('recip_total', reverse=True)

recipient_total.print_table(max_column_width=50)

purposes = skedb.group_by('purpose')

purpose_totals = purposes.aggregate([('purpose_total', agate.Sum('amount'))])

purpose_totals = purpose_totals.order_by('purpose_total', reverse=True)

purpose_totals.print_table(max_column_width=50)



