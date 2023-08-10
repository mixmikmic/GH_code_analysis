import re

word = r'\w+'

sentence = 'I am testing with Regex.'

search_result = re.search(word, sentence)

search_result

search_result.group()

match_result = re.match(word, sentence)

match_result

match_result.group()

re.findall(word, sentence)

capitalized_word = r'[A-Z]\w+'

search_result = re.search(capitalized_word, sentence)

search_result.group()

match_result = re.match(capitalized_word, sentence)

match_result

sentence_with_digits = 'The airport is 4,300 meters away, but I still hear 10 planes at night.'

numbers = r'\d+'

re.findall(numbers, sentence_with_digits)

thousands_numbers = '(\d+,\d+|\d+)'

re.findall(thousands_numbers, sentence_with_digits)

city_state = '(?P<city>[\w\s]+), (?P<state>[A-Z]{2})'

address = 'My House, 123 Main Street, Los Angeles, CA 90013'

match = re.finditer(city_state, address)

for city in match:
    print(city.group('city'))



