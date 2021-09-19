from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from scipy.stats.stats import zscore
from itertools import count
from sklearn.decomposition import PCA

import requests
import json


le = LabelEncoder()

app = Flask(__name__,template_folder='Templates')
model5 = pickle.load(open('model5.pkl','rb'))

@app.route('/', methods=['GET'])
def main():
    return render_template('delivery.html')

standard_to = StandardScaler()

@app.route("/predict",methods=['POST'])
def predict():
    # input

    origin = request.form["Restaurant"]
    destination = request.form["Location"]

    pred = getDeliveryTime(origin, destination, key="AIzaSyCwBkG7Rgsw_fxHajDmbDwpalqoHMlWZhQ")
    # model.predict x 
    # x = model5.predict([['address', 'location']])

    return render_template('delivery.html',pred = 'Estimated delivery time is {} minutes'.format(pred))

def getDeliveryTime(origin, destination, key):
    
    googleRequest = requests.get("https://maps.googleapis.com/maps/api/directions/json?origin=%s&destination=%s&mode=driving&key=%s" % (origin, destination, key)
 )
    data = json.loads(googleRequest.text)
    duration = data["routes"][0]["legs"][0]["duration"]["text"]
   
    return duration
  
    
    
    
if __name__ == '__main__':
    app.run(debug=True)
