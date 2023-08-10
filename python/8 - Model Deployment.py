import os
import glob
import pickle
from flask import Flask
from flask_restplus import Api, Resource, fields, reqparse
from werkzeug.contrib.fixers import ProxyFix
import pandas as pd
import numpy as np

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)
app.config.SWAGGER_UI_DOC_EXPANSION = 'full'
api = Api(app, version='1.0', title='House Prediction API',
    description='Expose ML Models as endpoints',
)

ns = api.namespace('api/v1', description='House prediction API')


def load_models() -> dict:
    model_filenames = glob.glob('./models/sf/*.pkl')
    models = {}
    for filename in model_filenames:
        # skip the simple linear model
        model_name = os.path.splitext(os.path.basename(filename))[0].lower()
        if 'simple_linear' in model_name:
            continue
        with open(filename, 'rb') as f:
            model = pickle.load(f)
            models[model_name] = model
    return models

MODELS = load_models()
AVAILABLE_PROPERTY_TYPES = pd.read_csv('./data/sf/data_clean_imputed.csv')['property_type'].unique()
AVAILABLE_ZIP_CODES = pd.read_csv('./data/sf/data_clean_imputed.csv')['postal_code'].unique()


def ensemble_prediction(input_data: pd.DataFrame) -> dict:
    predictions = {}
    avg = []
    for model_type in MODELS.keys():
        prediction = MODELS[f'{model_type}'].predict(input_data)
        prediction = float(prediction.squeeze())
        predictions[model_type] = prediction
        avg.append(prediction)
    predictions['ensemble'] = np.average(np.array(avg))
    return predictions

def predict(input_dict: dict, model_type: str) -> float:
    df_cols = pd.read_csv('./data/sf/data_clean_engineered.csv')
    features = [feature for feature in df_cols.columns if feature != 'price']
    df_input = pd.get_dummies(pd.DataFrame(data=[input_dict], columns=features).fillna(0))
    
    if 'ensemble' in model_type.lower():
        return ensemble_prediction(df_input)
    elif model_type.lower() not in AVAILABLE_MODELS:
        raise Exception(f"model type {model_type} not available. Available models: {AVAILABLE_MODELS} or ensemble")

    prediction = MODELS[f'{model_type}'].predict(df_input)
    return float(prediction.squeeze())

@ns.route('/prediction')
class Prediction(Resource):
    '''Prediction Endpoint'''
    @ns.param('bed', f'Number of bedrooms',
          type=int,
             required=True)
    @ns.param('bath', f'Number of bathrooms',
          type=int,
             required=True)
    @ns.param('sqft', f'Square footage',
          type=int,
             required=True)
    @ns.param('zipcode', f'Zip code (chocices: {AVAILABLE_ZIP_CODES})',
          type=str,
             required=True)
    @ns.param('property_type', f'Type of property (choices: {AVAILABLE_PROPERTY_TYPES})',
          type=str,
             required=True)
    @ns.param('model', f'Type of ML model to use (choices: {MODELS.keys()})',
          type=str,
             required=True,
             default='ensemble')
    def get(self):
        '''Get prediction'''
        try:
            parser = reqparse.RequestParser()
            parser.add_argument('bed', type=int, required=True)
            parser.add_argument('bath', type=int, required=True)
            parser.add_argument('sqft', type=int, required=True)
            parser.add_argument('zipcode', type=int, required=True)
            parser.add_argument('property_type', type=str, required=True)
            parser.add_argument('model', type=str, required=True)
            args = parser.parse_args()
            bed = args['bed']
            bath= args['bath']
            sqft= args['sqft']
            zipcode= args['zipcode']
            if zipcode not in AVAILABLE_ZIP_CODES:
                raise Exception(f'zipcode {zipcode} not available. Choices: {AVAILABLE_ZIP_CODES}')
            property_type= args['property_type']
            if property_type not in AVAILABLE_PROPERTY_TYPES:
                raise Exception(f'property_type {property_type} not available. Choices: {AVAILABLE_PROPERTY_TYPES}')
            model = args['model']
            input_dict = {
                'bed': bed,
                'bath': bath,
                'sqft': sqft,
                'postal_code_{}'.format(zipcode): 1,
                'property_type_{}'.format(property_type): 1,
                         }
            prediction = predict(input_dict=input_dict, model_type=model)

            return {'prediction': prediction}, 200
        except Exception as e:
            return {'error': str(e)}, 500

if __name__ == '__main__':
    load_models()
    app.run(host="0.0.0.0", port=int(os.environ['API_PORT']))



