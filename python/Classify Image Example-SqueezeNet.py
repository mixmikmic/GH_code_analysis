import turicreate as turi

url = "data/food_images"

data = turi.image_analysis.load_images(url)
data["foodType"] = data["path"].apply(lambda path: "Eggs" if "eggs" in path else "Soup")
data.save("egg_or_soup.sframe")
data.explore()

dataBuffer = turi.SFrame("egg_or_soup.sframe")

trainingBuffers, testingBuffers = dataBuffer.random_split(0.9)

model = turi.image_classifier.create(trainingBuffers, target="foodType", model="squeezenet_v1.1")

evaluations = model.evaluate(testingBuffers)
print evaluations["accuracy"]

model.save("egg_or_soup.model")

model.export_coreml("EggSoupClassifier.mlmodel")

