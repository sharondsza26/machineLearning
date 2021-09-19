# # Run only once 
from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition._pca import PCA
from scipy.stats.stats import zscore
from itertools import count


le = LabelEncoder()

app = Flask(__name__,template_folder='Templates')
model = pickle.load(open('model.pkl','rb'))

@app.route('/', methods=['GET'])
def main():
    return render_template('delivery.html')


standard_to = StandardScaler()

@app.route("/predict",methods=['POST'])
def predict():
    print('app.py works')
    # input
    Restaurant = request.form['Restaurant']
    print(Restaurant)
    Location = request.form['Location']
    Cuisines = request.form['Cuisines']
    Total_Cost = request.form['Total_Cost']
    Minimum_Order = request.form['Minimum_Order']
    Rating = request.form['Rating']
    Time_Taken_To_Cook = request.form['Time_Taken_To_Cook']
    No_Of_Items = request.form['No_Of_Items']

    df = pd.DataFrame([Restaurant, Location, Cuisines, Total_Cost, Minimum_Order, Rating, Time_Taken_To_Cook, No_Of_Items])

    # count = CountVectorizer()

    pca = PCA(n_components=22)
    data = pd.concat([pd.DataFrame(zscore(Cuisines.drop(['Cuisines'],axis=1)),columns=df),pd.DataFrame((count.transform(df['Cuisines']).todense()))],axis=1)
    x = pca.fit_transform(data)

    data['Delivery_Time'] = pd.DataFrame(model.predict(x))
    print(model.predict(x))

    pred = (model.predict(x))

    # pred = "hello"
    return render_template('delivery.html',pred = 'Estimated delivery time is {}'.format(pred))
    # return render_template('delivery.html')


if __name__ == '__main__':
    app.run(debug=True)

