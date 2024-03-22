from flask import Flask, render_template, request
import requests

app = Flask(__name__)

ml_server_url = 'http://ml_server:5001/predict'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def make_prediction():
    data = {
        'features': {
            'PassengerId': request.form['PassengerId'],
            'Pclass': request.form['Pclass'],
            'Name': request.form['Name'],
            'Sex': request.form['Sex'],
            'Age': request.form['Age'],
            'SibSp': request.form['SibSp'],
            'Parch': request.form['Parch'],
            'Ticket': request.form['Ticket'],
            'Fare': request.form['Fare'],
            'Cabin': request.form['Cabin'],
            'Embarked': request.form['Embarked'],
        },
        'model': request.form['model']
    }

    response = requests.post(ml_server_url, json=data)
    result = response.json()

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
