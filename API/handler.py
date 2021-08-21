import numpy as np
import pandas as pd
import pickle
from flask import Flask,request, Response
from VehicleInsurance.VehicleInsurance import VehicleInsurance
import os


model = pickle.load(open(r'Model\xgb_model.pkl','rb'))

app = Flask(__name__)

@app.route('/predict',methods=['POST'])

def health_insurance_predict():
    test_json = request.get_json()

    if test_json:
        if isinstance(test_json,dict):
            test_raw =pd.DataFrame(test_json, index=[0])
        
        else:
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())

        pipeline = VehicleInsurance()

        data_cleaned = pipeline.data_cleaning(test_raw)
        data_engineered = pipeline.feature_engineering(data_cleaned)
        data_prepared = pipeline.data_preparation(data_engineered)

        df_response = pipeline.get_prediction(model, test_raw, data_prepared)

        return df_response

    else:
        return Response('{}',status=200,mimetype='application/json')

if __name__ == '__main__':
    port = os.environ.get('PORT',5000)
    app.run(host = '127.0.0.1',port=port)