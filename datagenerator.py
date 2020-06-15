import pandas_datareader as web
import datetime
import math
import numpy as np
from datetime import datetime
from flask  import Flask,request, jsonify
def getStockData(stockSymbol,startDate,EndDate,intervalmargin) :
    x = datetime.fromtimestamp(int(EndDate))
    datetoday =x.strftime("%Y")+'-'+x.strftime("%m")+'-'+x.strftime("%d")
    y=datetime.fromtimestamp(int(startDate))
    date5yearsago =y.strftime("%Y")+'-'+y.strftime("%m")+'-'+y.strftime("%d")
    df= web.get_data_yahoo(stockSymbol,interval=intervalmargin, start=date5yearsago, end=datetoday)
    Close = (df.filter(['Close'])).values.tolist()
    High = (df.filter(['High'])).values.tolist()
    Low = (df.filter(['Low'])).values.tolist()
    Open = (df.filter(['Open'])).values.tolist()
    Time = (df.index).tolist()
    return jsonify({'Close' : Close, 'High' : High, 'Low' : Low ,'Open' : Open,'Date' : Time})