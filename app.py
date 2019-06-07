
#How to start the app in virtualenv

#go to the flask_api directory 
# type venv\Scripts\activate in terminal
# then python app.py



from flask import Flask, jsonify, request 
#pip3 install pandas
#pip3 install sklearn
#pip install keras 
#pip install tensorflow

import random, os, sys, requests
import pandas as pd
import numpy as np
import h5py

#importing required libraries to perform LSTM 
import tensorflow as tf
graph = tf.get_default_graph()

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Model, model_from_json, load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.neural_network import MLPRegressor
from keras.backend import clear_session
#Disabling the AVX CPU warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



#Correctly locate the database file
basedir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath('./models'))

#Init app
app = Flask(__name__)


#calling alphavantage endpoint to retrive data for desired symbol
API_KEY = 'DKEHNKFEN5UFI1F7'
@app.route('/predict/<string:symbol>', methods=['GET'])
def serve_predection(symbol):
    resp = requests.get('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol='+symbol+'&outputsize=full&apikey='+API_KEY)
    if resp.status_code != 200:
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

    #setting the index
    data.index = data.timestamp
    #sorting from dates
    data = data.sort_index(ascending= True, axis = 0)
    data.drop('timestamp', axis=1, inplace=True)
    
    
    
    #Here we only take the closing prices for last 60 days
    dataset = data.iloc[-425:, 1:2].values
    
    #feature Scaling 
    scaler = MinMaxScaler(feature_range = (0, 1))
    dataset = dataset.reshape(-1,1)
    dataset = dataset.astype(float)
    dataset = scaler.fit_transform(dataset)

    #reshapping the data 
    data_array = []
    for i in range(60, 425):
        data_array.append(dataset[i-60:i, 0])

    data_array = np.array(data_array)
    data_array = np.reshape(data_array, (data_array.shape[0], data_array.shape[1], 1))


    #Loading the Model
    json_file = open('LSTM_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("LSTM_model.h5")
    
    #in our computation graph
    #sending data_array to the model 
    prediction = loaded_model.predict(data_array)
    #performing tranform inverse on return data from the model
    prediction = scaler.inverse_transform(prediction)
    
    #prepearing to send JSON response back
    #converting predicition into datframe
    prediction = pd.DataFrame(data=prediction)
    
    serve_pred = pd.DataFrame(index=range(0,len(prediction)),columns=['Close'])
    serve_pred['Close'] = prediction[0]
    
    #This may a bug or an unsupported case in tensorflow keras backend: session is cached globally and is not cleared.
    #So we have to clear it manually otherwise there can be only one prediction per session
    clear_session()
    return jsonify(serve_pred.to_json()) 

   


#How to load json data to a python df

#Run Server
if __name__ == "__main__":
    app.run(debug=True)    
