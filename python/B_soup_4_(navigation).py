import requests
from BeautifulSoup import *

url = "https://hrantdavtyan.github.io/"

response = requests.get(url)
page = response.text
soup = BeautifulSoup(page)

# finding all labels
label_tags = soup.findAll('label')
print(label_tags)

# choosing our label of interest
email_label = label_tags[3]
print(email_label)

# navigating one tag forward and getting the text/string of it
email = email_label.findNext().text
print(email)

print(email_label.findAllNext())

email_sibling = email_label.findNextSibling().text
print(email_sibling)

email_parent = email_label.findParent()
print(email_parent)

email_parents = email_label.findParents()
print(email_parents)

email_parents[-1]

email_child = email_label.findChild()
print(email_child)

