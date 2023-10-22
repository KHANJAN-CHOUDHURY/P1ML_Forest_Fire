from flask import Flask, request, jsonify, render_template
# render_template: It helps finding the url of the html file
import pickle
import numpy
import pandas as pd
from sklearn.preprocessing import StandardScaler


application = Flask(__name__)
app = application

## My application should be able to interract with pickle files.
# Import the pickle files
ridge_model = pickle.load(open('models/Ridge.pkl','rb'))
standard_scaler = pickle.load(open('models/Ridge_scaler.pkl','rb'))

@app.route('/') # To access the starting page
def index():
    return render_template('index.html')
# render_template(): It will search for 'index.html' in the 'templates' folder. 

@app.route('/predictdata',methods=['GET','POST']) # To access the predicted value
def predict_datapoint():
    if request.method=="POST":
        # Taking input values for prediction through html file
        Temperature =  float(request.form.get('Temperature'))
        RH =  float(request.form.get('RH'))
        Ws =  float(request.form.get('Ws'))
        Rain =  float(request.form.get('Rain'))
        FFMC =  float(request.form.get('FFMC'))
        DMC =  float(request.form.get('DMC'))
        DC =  float(request.form.get('DC'))
        ISI =  float(request.form.get('ISI'))
        BUI =  float(request.form.get('BUI'))
        Classes =  float(request.form.get('Classes'))
        Region =  float(request.form.get('Region'))

        new_data_scaled = standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,DC,ISI,BUI,Classes,Region]])
        result = ridge_model.predict(new_data_scaled)#result: It will be in list formate
        
        return render_template('home.html',result=result[0])# Result is shown with help of 'home.html' file.
    else:
        return render_template('home.html')
""" 'POST': It is like sending some query to google search bar where will type some words which act as a
query. So in 'POST' we are sending some information to the server and retriving some information.
'GET': We only retriving information""" 
if __name__=="__main__":
    app.run(host="0.0.0.0") #app.run(host="0.0.0.0",port=8080): To change port number


"""When host="0.0.0.0" i.e. host address is 0.0.0.0 then program is connected to local machine where this 
program is running. Local machine will have local IP address. In the local machine having IP address
172.18.0.18:5000. This is not known to outsiders. So we will use url:'https://brown-translator-nerdr.pwskills.app'
and add port number 5000 at the end. So the new url will be:'https://brown-translator-nerdr.pwskills.app:5000'.
Now access 'index()' function we have to modify the url again with ("/") as show in '@app.route("/")'. So
the new url will be: 'https://brown-translator-nerdr.pwskills.app:5000/'"""