from flask import Flask, render_template, request
from joblib import load

app = Flask(__name__)

# Load model and scaler
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = load(os.path.join(BASE_DIR, '..', 'Training', 'floods.save'))
sc = load(os.path.join(BASE_DIR, '..', 'Training', 'transform.save'))


@app.route('/rainfall')
def rainfall_page():
    return render_template('rainfall_predict.html')

@app.route('/intro')
def intro():
    return render_template('image.html')

# Home page
@app.route('/')
def home():
    return render_template('home.html')

# Form page (GET only)
@app.route('/predict')
def predict_page():
    return render_template('predict.html')



@app.route('/data_predict_rainfall', methods=['POST'])
def data_predict_rainfall():

    cloud = float(request.form['cloud_cover'])
    annual = float(request.form['annual_rainfall'])
    janfeb = float(request.form['janfeb'])
    marchmay = float(request.form['marchmay'])
    junesept = float(request.form['junesept'])

    # ---- Feature conversion (proxy logic) ----

    temp = 35 - (cloud * 0.15)          # more clouds → lower temp
    Hum = 40 + (cloud * 0.6)            # more clouds → more humidity
    db = janfeb + marchmay + junesept   # drainage proxy
    ap = annual                         # direct mapping
    aal = (junesept / annual) * 100     # loss proxy

    data = [[temp, Hum, db, ap, aal]]

    prediction = model.predict(sc.transform(data))[0]

    if prediction == 0:
        return render_template('result.html', prediction='No possibility of severe flood')
    else:
        return render_template('result.html', prediction='Possibility of severe flood')


# Prediction logic (POST only)
@app.route('/data_predict', methods=['POST'])
def data_predict():
    temp = float(request.form['temp'])
    Hum  = float(request.form['Hum'])
    db   = float(request.form['db'])
    ap   = float(request.form['ap'])
    aal  = float(request.form['aal'])

    data = [[temp, Hum, db, ap, aal]]
    prediction = model.predict(sc.transform(data))[0]

    if prediction == 0:
        return render_template('result.html', prediction='No Possibility of severe flood')

    else:
        return render_template('result.html', prediction='Possibility of severe flood')


if __name__ == '__main__':
    app.run(debug=True)
