from tempfile import NamedTemporaryFile

f = NamedTemporaryFile('w+t')

# Write to the file, the output is the number of characters
f.write('Nobody lived on Deadweather but us and the pirates. It wasn’t hard to understand why.')

f.name

# Go to the top of the file
f.seek(0)

# Read the file
f.read()

f.close()

