somelist = list('SPAM')
somelist

somelist[0]

somelist[2]

'first={0}, last={1}'.format(somelist[0], somelist[-1])

'first={0[0]}, third={0[2]}, another={1}'.format(somelist, 'hello')

somedict = {'hello': 'world', 5: 3}
somedict

'first={0[hello]}, third={0[5]}, another={1}'.format(somedict, 'hello')

