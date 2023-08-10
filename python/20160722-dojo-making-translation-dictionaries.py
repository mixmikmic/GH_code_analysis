translation_dict = {
    ord("@"): "", ord("%"): "", ord("^"): " ",
    ord("?"): "", ord(">"): "", ord("*"): "",
    ord("("): "", ord(")"): "", ord("+"): "",
}

def foo(terms):
    return (term.translate(translation_dict) for term in terms)

test_terms = (
    ('@%?>*()+hello^world',),
)

for terms in test_terms:
    print(tuple(foo(terms)))

known_good_output = [tuple(foo(terms)) for terms in test_terms]

known_good_output

def test():
    assert known_good_output == [
        tuple(foo(terms)) for terms in test_terms]

test()

# Put each dictionary item is put on a separate line.

# Notice that one of the values is not ''.
# I did not notice the ' ' value in the original code.

translation_dict = {
    ord("@"): "",
    ord("%"): "",
    ord("^"): " ",
    ord("?"): "",
    ord(">"): "",
    ord("*"): "",
    ord("("): "",
    ord(")"): "",
    ord("+"): "",
}

test()

# Move the item with the different value to the top
# to draw attention to it.
#
# Which is easier to understand?

translation_dict = {
    ord("^"): " ",

    ord("@"): "",
    ord("%"): "",
    ord("?"): "",
    ord(">"): "",
    ord("*"): "",
    ord("("): "",
    ord(")"): "",
    ord("+"): "",
}

test()

# Factor out the ord() calls.
#
# Which is easier to understand?

translation_dict = {
    "^": " ",

    "@": "",
    "%": "",
    "?": "",
    ">": "",
    "*": "",
    "(": "",
    ")": "",
    "+": "",
}
translation_dict = {
    ord(key): value
    for key, value in translation_dict.items()
}

test()

# Use None value instead of ''
# to indicate characters to delete.
#
# Which is easier to understand?

translation_dict = {
    "^": " ",

    "@": None,
    "%": None,
    "?": None,
    ">": None,
    "*": None,
    "(": None,
    ")": None,
    "+": None,
}
translation_dict = {
    ord(key): value
    for key, value in translation_dict.items()
}

test()

# Factor out the redundant values
#
# Which is easier to understand?

deleteable_chars = '@%?>*()+'

translation_dict = {'^': ' '}
translation_dict.update({c: None for c in deleteable_chars})
translation_dict = {
    ord(key): value
    for key, value in translation_dict.items()
}

test()

# Swap the order.
#
# Which is easier to understand?

deleteable_chars = '@%?>*()+'
translation_dict = {c: None for c in deleteable_chars}
translation_dict.update({'^': ' '})

translation_dict = {
    ord(key): value
    for key, value in translation_dict.items()
}

test()

# Use dict(map()) to create dictionary or deleteable characters.
#
# I sure don't like this way.
# It is ugly and harder to understand.

deleteable_chars = '@%?>*()+'
translation_dict = dict(map(lambda x:(x, None), deleteable_chars))
translation_dict.update({'^': ' '})

translation_dict = {
    ord(key): value
    for key, value in translation_dict.items()
}

test()

# Use dict.fromkeys() to create initial directory.
#
# Which is easier to understand?

deleteable_chars = '@%?>*()+'
translation_dict = dict.fromkeys(deleteable_chars, None)
translation_dict.update({'^': ' '})

translation_dict = {
    ord(key): value
    for key, value in translation_dict.items()
}

test()

# Revert to using '' value instead of None
# to indicate characters to delete.
#
# Which is easier to understand?

deleteable_chars = '@%?>*()+'
translation_dict = dict.fromkeys(deleteable_chars, '')
translation_dict.update({'^': ' '})

translation_dict = {
    ord(key): value
    for key, value in translation_dict.items()
}

test()

