import pandas as pd

# Download changesets from osmcha as a csv file.
changesets = pd.read_csv('http://osmcha.mapbox.com/?date__gte=2017-01-01&date__lte=2017-04-30&is_suspect=False&is_whitelisted=All&harmful=None&checked=True&all_reason=True&render_csv=True')
changesets.head()

# Filter only changesets that have only one feature modified.
changesets = changesets[
    (changesets['create'] <= 0) &
    (changesets['modify'] == 1) &
    (changesets['delete'] <= 0)
]

# Remove duplicates by changeset ID.
print('Before removing duplicated: {}'.format(changesets.shape[0]))
changesets = changesets.drop_duplicates(subset='ID')
print('After removing duplicated: {}'.format(changesets.shape[0]))

# Write back filtered changesets to a file.
changesets.to_csv('../downloads/changesets-filtered.csv', index=False)

