import numpy as np
import pandas as pd
import pickle
from flask import Flask,request,render_template

app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict():
    
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    output = prediction
    
    return render_template('index.html',prediction_text = 'If Survived 1 or else 0 {}'.format(output))


    
if __name__ == '__main__':
    app.run(debug=True)
    
    
    
