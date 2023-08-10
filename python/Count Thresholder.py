import graphlab as gl

sf = gl.SFrame({'a' : [1,2,3,2,3], 'b' : [2,3,4,2,3]})
sf

thresholder = gl.feature_engineering.CountThresholder(threshold=2)

thresholder.fit_transform(sf)

sf = gl.SFrame({'categories': [['cat', 'mammal'],
                                         ['cat', 'mammal'],
                                         ['human', 'mammal'],
                                         ['seahawk', 'bird'],
                                         ['duck', 'bird'],
                                         ['seahawk', 'bird']]})
sf

thresholder = gl.feature_engineering.CountThresholder(threshold=2)
thresholder.fit_transform(sf)

sf = gl.SFrame({'attributes':
                [{'height':'tall', 'age': 'senior', 'weight': 'thin'},
                 {'height':'short', 'age': 'child', 'weight': 'thin'},
                 {'height':'giant', 'age': 'adult', 'weight': 'fat'},
                 {'height':'short', 'age': 'child', 'weight': 'thin'},
                 {'height':'tall', 'age': 'child', 'weight': 'fat'}]})
sf

thresholder = gl.feature_engineering.CountThresholder(threshold=2)
thresholder.fit_transform(sf)



