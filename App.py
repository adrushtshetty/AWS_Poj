import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import pickle
app = Flask(__name__)
model = pickle.load(open('Model.pklodel.pkl', 'rb'))

@app.route('/')
def home_page():
    return render_template("index.html")

@app.route('/crop_prediction')
def crop_prediction():
    return render_template("crop_pred.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/crop_prediction',methods=['POST'])
def predict():

    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model.predict(final_features)
    if prediction[0] == 20:
        output ="Rice"
    elif prediction[0] == 11:
        output ="Maize"
    elif prediction[0] == 3:
        output ="Chickpeas"
    elif prediction[0] == 9:
        output = "Kidneybeans"
    elif prediction[0] == 18:
        output ="Pigeonpeas"
    elif prediction[0] == 13:
        output ="Mothbeans"
    elif prediction[0] == 14:
        output ="Mungbeans"
    elif prediction[0] == 2:
        output ="Blackgram"
    elif prediction[0] == 10:
        output ="Lentil"
    elif prediction[0] == 19:
        output ="Pomegranate"
    elif prediction[0] == 1:
        output ="Bananas"
    elif prediction[0] == 12:
        output ="Mangoes"
    elif prediction[0] == 7:
        output ="Grapes"
    elif prediction[0] == 21:
        output ="Watermelons"
    elif prediction[0] == 15:
        output ="Muskmelons"
    elif prediction[0] == 0:
        output ="Apples"
    elif prediction[0] == 16:
        output ="Oranges"
    elif prediction[0] == 17:
        output ="Papayas"
    elif prediction[0] == 4:
        output ="Coconuts"
    elif prediction[0] == 6:
        output ="Cotton"
    elif prediction[0] == 8:
        output ="Jute"
    elif prediction[0] == 5:
        output ="Coffee"


    return render_template('crop_pred.html', prediction_text='{}'.format(output))

@app.route('/crop_requisites_1')
def bench_1():
    return render_template("bench_1.html")
@app.route('/crop_requisites_2')
def bench_2():
    return render_template("bench_2.html")
@app.route('/crop_requisites_3')
def bench_3():
    return render_template("bench_3.html")

app.run(debug=True)