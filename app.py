import numpy as np

from flask import Flask, render_template, request, jsonify
import joblib
from sklearn.preprocessing import StandardScaler

model_gbc = joblib.load('Gradient_Boost_Classifier.sav')
model_rfr = joblib.load('Random_Forest_Regressor.sav')

outcome_scaler = joblib.load('Outcome_Scaler.bin')
Pedigree_fn_scaler = joblib.load('Pedigree_fn_scaler.bin')

app = Flask(__name__)

@app.route('/',methods=['GET'])
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

	Age = int(request.form['patient_age'])
	glucose_level = int(request.form['glucose_level'])
	blood_pressure = float(request.form['blood_pressure'])
	insulin = float(request.form['insulin'])
	pregnancies = int(request.form['pregnancies'])
	skin_thickness = float(request.form['skin_thickness'])
	bmi = float(request.form['bmi'])

	# Error handling:
	if blood_pressure not in range(20,120):
		return render_template('results.html',prediction_text='Blood pressure not in range.',
				likelihood_text='Error.')

	if skin_thickness not in range(3,60):
		return render_template('results.html',prediction_text='Blood pressure not in range.',
				likelihood_text='Error.')

	if bmi not in range(5,70):
		return render_template('results.html',prediction_text='BMI not in range.',
				likelihood_text='Error.')


	# Recording the input features:-
	features = np.array([Age,glucose_level,blood_pressure,insulin,pregnancies,
	skin_thickness,bmi])


	outcome_predictions = model_gbc.predict(outcome_scaler.transform([features]))


	likelihood_predictions = model_rfr.predict(Pedigree_fn_scaler.transform([features]))[0]

	if outcome_predictions == 1:
		return render_template('results.html',prediction_text='The patient is suffering from diabetes.',
			likelihood_text='Diabetic susceptibility percentage: {}'.format(round(likelihood_predictions*100)))
	else:
		return render_template('results.html',
				prediction_text='The patient is not suffering from diabetes.',
				likelihood_text="Diabetic susceptibility percentage: {}%".format(round(likelihood_predictions*100)))


if __name__ == '__main__':
	app.run(debug=True)