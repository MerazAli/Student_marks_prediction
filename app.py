from flask import Flask,render_template,request
import numpy as np
import pickle
import joblib     


app=Flask(__name__)

model = joblib.load('student_marks_predictor.pkl')

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    input_feature =[float(x) for x in request.form.values()]
    featuer_values =np.array(input_feature)
    
    output = model.predict([featuer_values])[0][0].round(2)
    
    return render_template('index.html',prediction_text='you will get [{}%] marks ,when you study [] hour per day')




if __name__== "__main__":
    app.run(debug=True)