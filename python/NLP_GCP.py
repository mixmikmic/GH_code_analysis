# Imports the Google Cloud client library
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types

text = "LeBron James plays for the Cleveland Cavaliers."
client = language.LanguageServiceClient()
document = types.Document(
        content=text,
        type=enums.Document.Type.PLAIN_TEXT)
entities = client.analyze_entities(document).entities

entities



