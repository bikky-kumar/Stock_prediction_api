from flask import Flask, jsonify, request 
#pip3 install pandas
#pip3 install sklearn
import random, os, requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))


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
    #converting JSON result to dataframe  
    df = pd.DataFrame.from_dict(dataForAllDays)
    #transposing data
    df_transposed = df.transpose()
    #creating a new df
    data = pd.DataFrame(index=range(0,len(df_transposed)),columns=['Date', 'Close'])
    data['Date'] = df_transposed.index
    df_transposed['4. close'].apply(pd.to_numeric)
    for i in range(0,len(data)):
     data['Close'][i] = df_transposed['4. close'][i]

    data['timestamp'] = pd.to_datetime(data.Date, format='%Y-%m-%d')

    #sorting the data
    #data = df.sort_index(ascending=True, axis=0)

    #separating dates, month and year
    data['year'] = data['timestamp'].dt.year
    data['month'] = data['timestamp'].dt.month
    data['dayofweek'] = data['timestamp'].dt.dayofweek

    #Deleting values not required
    data = data.drop('timestamp', axis=1)
    data = data.drop('Date', axis=1)

    print(data)
    return jsonify(dataForAllDays)   


#How to load json data to a python df

#Run Server
if __name__ == "__main__":
    app.run(debug=True)    
