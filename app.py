import numpy as np
from flask import Flask, request, render_template
import pickle
import joblib
import sklearn
import numpy

app = Flask(__name__)

model = joblib.load("./models/real_estate_model.pkl")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    features = [np.array(int_features)]
    prediction = model.predict(features)
    output = round(prediction[0], 2)
    return render_template('index.html', prediction_text='Price per sq. meter : {}'.format(output))


print(pickle.format_version)

if __name__ == "__main__":
    app.run()
