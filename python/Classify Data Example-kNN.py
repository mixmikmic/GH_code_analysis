import turicreate as turi

dataBuffer = turi.SFrame({
    "pet_types": ["cat", "dog", "wolf", "cat", "wolf", "dog"],
    "eyes":      [0.23,  0.64,  0.89,   0.26,  0.93,   0.66 ],
    "nose":      [0.11,  0.68,  0.78,   0.08,  0.74,   0.57 ],
    "head":      [0.34,  0.47,  0.66,   0.37,  0.68,   0.45 ]
})
print dataBuffer

classifier = turi.nearest_neighbor_classifier.create(dataBuffer, target="pet_types")

userinput = turi.SFrame({
    "eyes":      [0.32],
    "nose":      [0.20],
    "head":      [0.42]
}) 

result = classifier.predict(userinput, max_neighbors=2)
print result

