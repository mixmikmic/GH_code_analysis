# Importing Optimus
import optimus as op
#Importing utilities
tools = op.Utilities()

# Creating DF with Optimus
data = [('Japan', 'Tokyo', 37800000),('USA', 'New York', 19795791),('France', 'Paris', 12341418),
              ('Spain','Madrid',6489162)]
df = tools.create_data_frame(data, ["country", "city", "population"])

# Instantiating transformer
transformer = op.DataFrameTransformer(df)

# Show DF
transformer.show()

# Indexing columns 'city" and 'country'
transformer.string_to_index(["city", "country"])

# Show indexed DF
transformer.show()

# Instantiating transformer
transformer = op.DataFrameTransformer(df)

# Show DF
transformer.show()

# Indexing columns 'city" and 'country'
transformer.string_to_index(["city", "country"])

# Show indexed DF
transformer.show()

# Going back to strings from index
transformer.index_to_string(["country_index"])

# Show DF with column "county_index" back to string
transformer.show()

# Creating DataFrame
data = [
(0, "a"),
(1, "b"),
(2, "c"),
(3, "a"),
(4, "a"),
(5, "c")
]
df = tools.create_data_frame(data,["id", "category"])

# Instantiating the transformer
transformer = op.DataFrameTransformer(df)

# One Hot Encoding
transformer.one_hot_encoder(["id"])

# Show encoded dataframe
transformer.show()

# Creating DataFrame
data = [
(0, 1.0, 3.0),
(2, 2.0, 5.0)
]

df = tools.create_data_frame(data,["id", "v1", "v2"])

# Instantiating the transformer
transformer = op.DataFrameTransformer(df)

transformer.show()

transformer.sql("SELECT *, (v1 + v2) AS v3, (v1 * v2) AS v4 FROM __THIS__")

transformer.show()

# Import Vectors
from pyspark.ml.linalg import Vectors

# Creating DataFrame
data = [(0, 18, 1.0, Vectors.dense([0.0, 10.0, 0.5]), 1.0)]

df = tools.create_data_frame(data,["id", "hour", "mobile", "user_features", "clicked"])

df.show()

# Instantiating the transformer
transformer = op.DataFrameTransformer(df)

# Assemble features
transformer.vector_assembler(["hour", "mobile", "user_features"])


# Show assembled df
print("Assembled columns 'hour', 'mobile', 'user_features' to vector column 'features'")
transformer.df.select("features", "clicked").show(truncate=False)

# Importing Optimus
import optimus as op
#Importing utilities
tools = op.Utilities()
# Import Vectors
from pyspark.ml.linalg import Vectors

data = [
(0, Vectors.dense([1.0, 0.5, -1.0]),),
(1, Vectors.dense([2.0, 1.0, 1.0]),),
(2, Vectors.dense([4.0, 10.0, 2.0]),)
]

df = tools.create_data_frame(data,["id", "features"])

transformer = op.DataFrameTransformer(df)

transformer.show()

transformer.normalizer(["features"], p=1.0).show(truncate=False)



