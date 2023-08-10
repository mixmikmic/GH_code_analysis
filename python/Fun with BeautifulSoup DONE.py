from bs4 import BeautifulSoup

with open('table_example.html', 'rb') as infile:
    example = infile.read()

soup = BeautifulSoup(example, 'html.parser')

print(soup.prettify())

table = soup.find('table')

for row in table.find_all('tr'):
    cell_holder = []
    for cell in row.find_all(['th', 'td']):
        cell_holder.append(cell.text)
    print(', '.join(cell_holder))

