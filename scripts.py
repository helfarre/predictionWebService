#!/usr/bin/env python
# coding: utf-8

# In[51]:


import datetime
import math
import pandas_datareader.data as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM,Dropout
from json import JSONEncoder

def evaluate_model(stockSymbol):
    x = datetime.datetime.now()
    datetoday =x.strftime("%Y")+'-'+x.strftime("%m")+'-'+x.strftime("%d")
    y= datetime.datetime.now() - datetime.timedelta(days=5*365)
    date5yearsago =y.strftime("%Y")+'-'+y.strftime("%m")+'-'+y.strftime("%d")
    df= web.DataReader(stockSymbol, data_source='yahoo', start=date5yearsago, end=datetoday)
    data = df.filter(['Close'])
    dataset = data.values
    training_data_len= math.ceil(len(dataset))
    train_data_unscaled = dataset[0:training_data_len, :]
    Scaler = MinMaxScaler(feature_range=(0,1))
    train_data= Scaler.fit_transform(train_data_unscaled)
    x_train = []
    y_train = []
    for i in range(30, len(train_data)-5):
      x_train.append(train_data[i-30:i,0])
      y_train.append(train_data[i:i+5,0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train= np .reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
    model = Sequential()
    model.add(LSTM(100,activation='relu',return_sequences=True,input_shape=(x_train.shape[1],1)))
    model.add(LSTM(50,activation='relu',return_sequences=False))
    model.add(Dense(5))
    model.compile(optimizer='adam', loss='mean_squared_error')
    history = model.fit(x_train,y_train,batch_size=50,epochs=10)
    model.save(stockSymbol+".h5")
    print("Saved model to disk")



def predict(StockName, model):
    Scaler = MinMaxScaler(feature_range=(0,1))
    x = datetime.datetime.now()
    datetoday =x.strftime("%Y")+'-'+x.strftime("%m")+'-'+x.strftime("%d")
    y= datetime.datetime.now() - datetime.timedelta(days= 60)
    date3day =y.strftime("%Y")+'-'+y.strftime("%m")+'-'+y.strftime("%d")
    df= web.DataReader(StockName, data_source='yahoo', start=date3day, end=datetoday)
    data = df.filter(['Close'])
    data= data.tail(30)
    predict_data_input= Scaler.fit_transform(data)
    x_test= np.reshape(predict_data_input,(predict_data_input.shape[1],predict_data_input.shape[0],1))
    predictions = np.empty([len(x_test), 5], dtype=np.float32)
    BATCH_INDICES = np.arange(start=0, stop=len(x_test), step=8) 
    BATCH_INDICES = np.append(BATCH_INDICES, len(x_test))  
    for index in np.arange(len(BATCH_INDICES) - 1):
        batch_start = BATCH_INDICES[index]  
        batch_end = BATCH_INDICES[index + 1]  
        predictions[batch_start:batch_end] = model.predict_on_batch(x_test[batch_start:batch_end])
    predictions = Scaler.inverse_transform(predictions)
    return np.asarray(predictions[0])

def getLatestPrice(StockName):
    Scaler = MinMaxScaler(feature_range=(0,1))
    x = datetime.datetime.now()
    datetoday =x.strftime("%Y")+'-'+x.strftime("%m")+'-'+x.strftime("%d")
    y= datetime.datetime.now() - datetime.timedelta(days= 5)
    date3day =y.strftime("%Y")+'-'+y.strftime("%m")+'-'+y.strftime("%d")
    df= web.DataReader(StockName, data_source='yahoo', start=date3day, end=datetoday)
    data = df.filter(['Close'])
    data =np.asarray(data)
    C = data.item(data.size-1)
    tf.keras.backend.clear_session()
    return C
    # return predictions
    
class PredictionResult:
    predictions = 0
    todayPrice = 0

class ObjectEncoder(JSONEncoder):
        def default(self, o):
            return o.__dict__