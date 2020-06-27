from flask  import Flask,request, jsonify
from scripts import evaluate_model, predict , getLatestPrice, PredictionResult, ObjectEncoder
from datagenerator import getStockData
from tensorflow.keras.models import load_model
import os
from os import path
from flask_cors import CORS
from json import JSONEncoder
import json
from flask import Response
import logging

# init app 
app = Flask(__name__)
cors = CORS(app, resources={"*": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'
logger = logging.getLogger()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - "
                      "%(name)s:%(lineno)d [-] %(funcName)s- %(message)s")
logger.setLevel(logging.INFO)


basedir = os.path.abspath(os.path.dirname(__file__))

@app.route('/getPrice/<stock>',methods=['GET'])
def getPrive(stock):
    latestprice = float(getLatestPrice(stock))
    return jsonify({'price' : latestprice})

@app.route('/getStockHistory/<stock>/<startDate>/<endDate>/<intervalmargin>',methods=['GET'])
def getStockPrice(stock,startDate,endDate,intervalmargin):
    return getStockData(stock,startDate,endDate,intervalmargin)

@app.route('/compile/<stock>',methods=['GET'])
def compile(stock):
    evaluate_model(stock)
    return jsonify({'msg' : 'Model was creation succesfuly'})

    
# Run Server
@app.route('/predict/<stock>',methods=['GET'])
def predictee(stock):
    if path.exists(stock+'.h5') :
        model = load_model(stock+'.h5')
    else :
        evaluate_model(stock)
        model = load_model(stock+'.h5') 
    predictions = predict(stock,model).tolist()
    latestprice = float(getLatestPrice(stock))
    obj = PredictionResult()
    obj.predictions=predictions
    obj.todayPrice=latestprice
    employeeJSONData = json.dumps(obj, indent=4, cls=ObjectEncoder)
    r = Response(response=employeeJSONData, status=200, mimetype="application/json")
    r.headers["Content-Type"] = "application/json; charset=utf-8"
    return r

    
if __name__ == '__main__':
    app.run(debug=True)