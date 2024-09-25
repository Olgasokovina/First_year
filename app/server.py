import pickle

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from func_for_server import *
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV

with open(r'best_model.pkl', 'rb') as pkl_file:
    model = pickle.load(pkl_file)
with open(r'automl.pkl', 'rb') as pkl_file:
    automl = pickle.load(pkl_file)
with open(r'selector_RFECV.pkl', 'rb') as pkl_file:
    selector_RFECV = pickle.load(pkl_file)


app = Flask(__name__)


@app.route('/')
def index():
    return "Тестовое сообщение. Сервер запущен!"


@app.route('/predict', methods=['POST'])
def predict():

    features = data_preprocessing(pd.read_json(request.json, orient='table'))
    prediction_autoML = float(automl.predict(features).data[0, 0])

    temp = selector_RFECV.get_feature_names_out()
    prediction = model.predict(features[temp])

    return jsonify({
        'prediction_autoML': prediction_autoML,
        'prediction': prediction[0],
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
