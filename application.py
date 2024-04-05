from flask import Flask,request,render_template,jsonify
import pickle
import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

# step1 -> import the model 
scaler_model = pickle.load(open('models/scaler_final.pkl' ,'rb'))
ridge_model = pickle.load(open('models/ridge_final.pkl','rb'))

# home page 
@app.route('/')
def index():
    return render_template('home.html')

# predict korte hbe
# GET -> directly url e /predict likhe
# POST -> form e submit korey 

@app.route('/predict' , methods=['GET','POST']) 
def predict_datapoint():

    if request.method == 'POST':

        #store all the input 
        # same name and same order e store korte hoi 
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes =float( request.form.get('Classes') )
        Region = float (request.form.get('Region'))

        # scale down all the input 
        scale_data = scaler_model.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])

        #predict the output 
        predict_output = ridge_model.predict(scale_data)

        # show the result 
        return render_template('home.html' , result=predict_output[0])


    else:
        #jodi url er through ase then amra form take render korbo 
        return render_template('home.html')



if __name__=="__main__":
    app.run(host="0.0.0.0")
