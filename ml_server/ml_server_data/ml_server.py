from flask import Flask, request, jsonify
from pymongo import MongoClient
import mysql.connector
import pandas as pd

import os
import sys
import json

# Add the path to machine_learning to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
my_package_path = os.path.abspath(os.path.join(current_dir, 'my_package'))

sys.path.append(my_package_path)

from machine_learning import data_preprocess as dp

app = Flask(__name__)

# MongoDB Configuration
mongo_client = MongoClient('mongodb://admin:password@mongadb:27017/')
mongo_db = mongo_client['ml_data']

# MySQL Configuration
mysql_connection = mysql.connector.connect(
    host='mysql',
    user='root',
    password='root_password',
    database='ml_data'
)

cursor = mysql_connection.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    input_data JSON,
    model_type JSON,
    prediction JSON,
    prediction_prob JSON
)
""")

job_files = {
    'data_parser': 'data_parser_v1',
    'null_imputation': 'null_imputation_v1',
    'one_hot_encoder': 'one_hot_encoder_v1',
    'data_filter': 'data_filter_v1',
    'model_lr': 'logistic_regression_model_v1',
    'model_rf': 'random_forest_classification_model_v1',
    'model_SVC': 'SVC_model_v1',
    'model_xgb': 'xgboost_classifier_model_v1'
}

loaded_jobs = {}

current_dir = os.path.dirname(os.path.abspath(__file__))
model_folder = os.path.abspath(os.path.join(current_dir, 'models'))

for job_name, job_file in job_files.items():
    job_path = os.path.join(model_folder, job_file)
    loaded_job = dp.my_joblib.load_job(job_path, '')
    loaded_jobs[job_name] = loaded_job

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    df = pd.json_normalize(data['features'])
    debug_file_save(df,'df.csv')

    df1 = loaded_jobs['data_parser'].apply_data_parser(df)
    debug_file_save(df1,'df1.csv')

    df2 = loaded_jobs['null_imputation'].apply_null_imputation(df1)
    debug_file_save(df2,'df2.csv')

    df3 = loaded_jobs['one_hot_encoder'].apply_my_one_hot_encoder(df2)
    debug_file_save(df3,'df3.csv')

    df4 = loaded_jobs['data_filter'].apply_data_filter(df3)
    debug_file_save(df4,'df4.csv')

    model_type = data.get('model') 
    selected_model = loaded_jobs[model_type]

    prediction = selected_model.predict(df4)
    prediction_prob = selected_model.predict_proba(df4)

    prediction_data = {
        'input_data': data,
        'model_type': model_type,
        'prediction': prediction.tolist(),
        'prediction_prob': prediction_prob.tolist()
    }

    insert_prediction_to_mongodb(prediction_data)
    insert_prediction_to_mysql(prediction_data)


    prediction_prob_str = [str(item) for item in prediction_data['prediction_prob']]
    prediction_data_str = {
        'input_data': prediction_data['input_data'],
        'model_type': prediction_data['model_type'],
        'prediction': prediction_data['prediction'],
        'prediction_prob': prediction_prob_str
    }

    return jsonify(prediction_data_str)
    
def insert_prediction_to_mongodb(data):
    try:
        mongo_db.ml_data.insert_one(data)
    except Exception as e:
        print("An error occurred while inserting data into MongoDB:", e)

def insert_prediction_to_mysql(data):
    try:
        query = "INSERT INTO predictions (input_data, model_type, prediction, prediction_prob) VALUES (%s, %s, %s, %s)"
        values = (json.dumps(data['input_data']), data['model_type'], json.dumps(data['prediction']), json.dumps(data['prediction_prob']))
        cursor.execute(query, values)
        mysql_connection.commit()
    except Exception as e:
        print("An error occurred while inserting data into MySQL:", e)


def debug_file_save(data,file_name):
    save_folder = '/app/data' 
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    csv_file_path = os.path.join(save_folder, file_name)
    data.to_csv(csv_file_path, index=False)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)

