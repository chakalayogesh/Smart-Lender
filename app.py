from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")

model = joblib.load(MODEL_PATH)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    gender = int(request.form["Gender"])
    married = int(request.form["Married"])
    dependents = int(request.form["Dependents"])
    education = int(request.form["Education"])
    self_employed = int(request.form["Self_Employed"])
    income = float(request.form["ApplicantIncome"])
    coincome = float(request.form["CoapplicantIncome"])
    loanamount = float(request.form["LoanAmount"])
    term = float(request.form["Loan_Amount_Term"])
    credit = float(request.form["Credit_History"])
    area = int(request.form["Property_Area"])

    data = pd.DataFrame([[gender, married, dependents, education,
                          self_employed, income, coincome,
                          loanamount, term, credit, area]])

    prediction = model.predict(data)[0]

    result = "Loan Approved" if prediction == 1 else "Loan Rejected"

    return render_template("result.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)