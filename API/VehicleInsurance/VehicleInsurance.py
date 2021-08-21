import numpy as np 
import pandas as pd 
import pickle

class VehicleInsurance:

    def __init__(self):
        self.annual_premium_scaler =            pickle.load( open(r'Parameters\annual_premium_scaler.pkl','rb'))
        self.age_scaler =                       pickle.load( open(r'Parameters\age_scaler.pkl','rb'))
        self.vintage_scaler =                   pickle.load( open(r'Parameters\vintage_scaler.pkl','rb'))
        self.region_code_encoder =              pickle.load( open(r'Parameters\region_code_encoder.pkl','rb'))
        self.policy_sales_channel_encoder =     pickle.load( open(r'Parameters\policy_sales_channel_encoder.pkl','rb'))

    def data_cleaning(self,data):

        lower = lambda x: x.lower()
        data.columns = list(map(lower,data.columns))
        data['region_code'] = data['region_code'].astype('int64')
        data['policy_sales_channel'] = data['policy_sales_channel'].astype('int64')

        data = data.drop('id',axis=1)
        data.index = data.index + 1

        return data

    def feature_engineering(self,data):
        rule = lambda x: 1 if (x['vehicle_damage'] == 'Yes') & (x['previously_insured'] == 0) else 0 
        data['no_insured_damage'] = data.apply(rule,axis=1)

        return data

    def data_preparation(self,data):

        vehicle_damage_dict = {'No' : 0, 'Yes' : 1}
        data['vehicle_damage'] = data['vehicle_damage'].map(vehicle_damage_dict)

        data['annual_premium'] = self.annual_premium_scaler.transform(data[['annual_premium']].values)
        data['age'] = self.age_scaler.transform(data[['age']].values)
        data['vintage'] = self.vintage_scaler.transform(data[['vintage']].values)
        
        data['region_code'] = data['region_code'].map(self.region_code_encoder)
        data['policy_sales_channel'] = data['policy_sales_channel'].map(self.policy_sales_channel_encoder)

        selected_columns = ['vintage','annual_premium','age','region_code','policy_sales_channel','no_insured_damage','vehicle_damage','previously_insured']

        return data[selected_columns]

    def get_prediction(self,model,original_data,test_data):

        pred = model.predict_proba(test_data)
        original_data['score'] = pred[:,1].tolist()

        return original_data.to_json(orient='records', date_format='iso')