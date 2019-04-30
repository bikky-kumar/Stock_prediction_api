from flask import Flask, jsonify, request 

import random, os, requests

from quotes import funny_quotes 


#How to start the app in virtualenv

#go to the flask_api directory 
# type venv\Scripts\activate in terminal
# then python app.py


#Init app
app = Flask(__name__)

#Correctly locate the database file
basedir = os.path.abspath(os.path.dirname(__file__))


#calling alphavantage endpoint to retrive data for desired symbol
API_KEY = 'DKEHNKFEN5UFI1F7'
@app.route('/predict/<string:symbol>', methods=['GET'])
def serve_predection(symbol):
    resp = requests.get('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol='+symbol+'&outputsize=full&apikey='+API_KEY)
    if resp.status_code != 200:
        # This means something went wrong.
        print(resp.status_code)
    result = resp.json()
    dataForAllDays = result['Time Series (Daily)']    
    return jsonify(dataForAllDays)   


#How to load json data to a python df

#Run Server
if __name__ == "__main__":
    app.run(debug=True)    
