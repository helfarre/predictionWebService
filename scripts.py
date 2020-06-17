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
#     print(df)
#     plt.figure(figsize=(16,8))
#     plt.title('close price history')
#     plt.plot(df['Close'])
#     plt.xlabel('Date',fontsize=12)
#     plt.ylabel('close price usd ($)', fontsize=18)
#     plt.show()
    data = df.filter(['Close'])
    dataset = data.values
    # print(dataset)
    training_data_len= math.ceil(len(dataset) * 0.9)
    train_data_unscaled = dataset[0:training_data_len, :]
#     train_data_unscaled
    Scaler = MinMaxScaler(feature_range=(0,1))
    train_data= Scaler.fit_transform(train_data_unscaled)
#     train_data
    x_train = []
    y_train = []
    for i in range(30, len(train_data)-5):
      x_train.append(train_data[i-30:i,0])
      y_train.append(train_data[i:i+5,0])
    x_train, y_train = np.array(x_train), np.array(y_train)
#     y_train.shape

#     print(x_train[5])
#     print(y_train[5])
    x_train= np .reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
#     x_train.shape

    #Construction du modele
    model = Sequential()
    model.add(LSTM(90,activation='relu',return_sequences=True,input_shape=(x_train.shape[1],1)))
    model.add(LSTM(45,activation='relu',return_sequences=False))

#Notre modele retourne à la fin les 5 valeurs des 5 futures journées
    model.add(Dense(5))
    model.compile(optimizer='adam', loss='mean_squared_error',metrics=['accuracy'])
#     model.summary()
    history = model.fit(x_train,y_train,batch_size=50,epochs=110,validation_split=0.1)
#     plt.figure(figsize=(16,8))
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#     plt.title('model train vs validation loss')
#     plt.ylabel('loss')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'validation'], loc='upper right')
#     plt.show()
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
    # print(data)
    predict_data_input= Scaler.fit_transform(data)
#     predict_data_input= np.array(predict_data_input)
    x_test= np.reshape(predict_data_input,(predict_data_input.shape[1],predict_data_input.shape[0],1))
    predictions = np.empty([len(x_test), 5], dtype=np.float32)
    BATCH_INDICES = np.arange(start=0, stop=len(x_test), step=8)  # row indices of batches
    BATCH_INDICES = np.append(BATCH_INDICES, len(x_test))  # add final batch_end row
    for index in np.arange(len(BATCH_INDICES) - 1):
        batch_start = BATCH_INDICES[index]  # first row of the batch
        batch_end = BATCH_INDICES[index + 1]  # last row of the batch
        predictions[batch_start:batch_end] = model.predict_on_batch(x_test[batch_start:batch_end])
    #Inversion de normalisation ( Revenir aux valeurs originale)
    predictions = Scaler.inverse_transform(predictions)
    # print(predictions) 
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