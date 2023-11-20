from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load('heartprediction.pkl')

@app.route('/')
def home():
    return render_template('HeartPrediction.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input values from the form and convert them to float
        age = float(request.form['age'])
        sex = float(request.form['sex'])
        cp = float(request.form['cp'])
        trestbps = float(request.form['trestbps'])
        chol = float(request.form['chol'])
        fbs = float(request.form['fbs'])
        restecg = float(request.form['restecg'])
        thalach = float(request.form['thalach'])
        exang = float(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = float(request.form['slope'])
        ca = float(request.form['ca'])
        thal = float(request.form['thal'])

        # Make a prediction using the loaded model
        prediction = model.predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

        return render_template('HeartPrediction.html', prediction=prediction[0])

    except ValueError:
        error_message = "Invalid input. Please enter numeric values for all input fields."
        return render_template('HeartPrediction.html', error=error_message)

if __name__ == '__main__':
    app.run(debug=True)
