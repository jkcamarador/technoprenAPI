from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.simplefilter("ignore")

symptoms = np.array(['itching', 'skin rash', 'nodal skin eruptions', 'dischromic patches', 'continuous sneezing', 'shivering',
                      'chills', 'watering from eyes', 'stomach pain', 'acidity', 'ulcers on tongue', 'vomiting', 'cough', 'chest pain',
                      'yellowish skin', 'nausea', 'loss of appetite', 'abdominal pain', 'yellowing of eyes', 'burning micturition', 'spotting urination',
                      'passage of gases', 'internal itching', 'indigestion', 'muscle wasting', 'patches in throat', 'high fever', 'extra marital contacts',
                      'fatigue', 'weight loss', 'restlessness', 'lethargy', 'irregular sugar level', 'blurred and distorted vision', 'obesity', 'excessive hunger',
                      'increased appetite', 'polyuria', 'sunken eyes', 'dehydration', 'diarrhoea', 'breathlessness', 'family history', 'mucoid sputum', 'headache',
                      'dizziness', 'loss of balance', 'lack of concentration', 'stiff neck', 'depression', 'irritability', 'visual disturbances', 'back pain',
                      'weakness in limbs', 'neck pain', 'weakness of one body side', 'altered sensorium', 'dark urine', 'sweating', 'muscle pain', 'mild fever',
                      'swelled lymph nodes', 'malaise', 'red spots over body', 'joint pain', 'pain behind the eyes', 'constipation', 'toxic look (typhos)',
                      'belly pain', 'yellow urine', 'receiving blood transfusion', 'receiving unsterile injections', 'coma', 'stomach bleeding', 'acute liver failure',
                      'swelling of stomach', 'distention of abdomen', 'history of alcohol consumption', 'fluid overload', 'phlegm', 'blood in sputum', 'throat irritation',
                      'redness of eyes', 'sinus pressure', 'runny nose', 'congestion', 'loss of smell', 'fast heart rate', 'rusty sputum', 'pain during bowel movements',
                      'pain in anal region', 'bloody stool', 'irritation in anus', 'cramps', 'bruising', 'swollen legs', 'swollen blood vessels', 'prominent veins on calf',
                      'weight gain', 'cold hands and feets', 'mood swings', 'puffy face and eyes', 'enlarged thyroid', 'brittle nails', 'swollen extremeties', 'abnormal menstruation',
                      'muscle weakness', 'anxiety', 'slurred speech', 'palpitations', 'drying and tingling lips', 'knee pain', 'hip joint pain', 'swelling joints', 'painful walking',
                      'movement stiffness', 'spinning movements', 'unsteadiness', 'pus filled pimples', 'blackheads', 'scurring', 'bladder discomfort', 'foul smell ofurine',
                      'continuous feel of urine', 'skin peeling', 'silver like dusting', 'small dents in nails', 'inflammatory nails', 'blister', 'red sore around nose', 'yellow crust ooze'])

df = pd.read_csv('precautions_description.csv')
model = joblib.load('model.pkl')

app = Flask(__name__)
@app.route('/')
def index():
    return "TECHNOPREN API"

@app.route('/predict',methods=['POST'])
def predict():
    patient_data = [0] * 131

    symps = request.form.get('symptopms')

    for symp in symps.split(','):
        patient_data[int(np.where(symptoms == symp)[0])] = 1

    prediction = model.predict([patient_data])[0]
    mask = df['Disease'] == str(prediction).strip()
    result = df.loc[mask, ['Description', 'Precautions']]

    description = result['Description'].iloc[0]
    precaution = result['Precautions'].iloc[0]

    return jsonify(
        disease=str(prediction).strip(),
        description=description,
        precaution=precaution
    )

if __name__ == '__main__':
    app.run(debug=False)


