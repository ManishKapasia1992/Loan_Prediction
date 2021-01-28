from flask import Flask, render_template, request
import pickle
import numpy as np

# loads the classifier model

filename = r'C:\Users\admin\Desktop\lr_classifier_model.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        gender = int(request.form['Gender'])
        married = int(request.form['Married'])
        dependents = int(request.form['Dependents'])
        education = int(request.form['Education'])
        self_employed = int(request.form['Self_Employed'])
        applicant_income = int(request.form['ApplicantIncome'])
        coapplicant_income = int(request.form['CoapplicantIncome'])
        loan_amount = int(request.form['LoanAmount'])
        loan_amount_term = int(request.form['LoanAmount_Term'])
        credit_history = int(request.form['Credit_History'])
        property_area = int(request.form['Property_Area'])

        data = np.array([[gender, married, dependents, education, self_employed, applicant_income,
                         coapplicant_income, loan_amount, loan_amount_term, credit_history, property_area]])

        my_prediction = classifier.predict(data)

        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
    app.run(debug=True)