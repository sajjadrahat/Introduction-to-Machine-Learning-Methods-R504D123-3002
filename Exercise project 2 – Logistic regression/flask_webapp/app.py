from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import joblib
import os

# Load model, scaler, and column names here
model = joblib.load(os.path.join(os.path.dirname(__file__), 'model.pkl'))
scaler = joblib.load(os.path.join(os.path.dirname(__file__), 'scaler.pkl'))
columns = joblib.load(os.path.join(os.path.dirname(__file__), 'columns.pkl'))

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            input_data = [float(request.form[col]) for col in columns]
            df = pd.DataFrame([input_data], columns=columns)

            scaled = scaler.transform(df)
            prediction = model.predict(scaled)[0]
            prob = model.predict_proba(scaled)[0][1]

            result = "APPROVED" if prediction == 1 else "NOT APPROVED"
            return render_template('index.html', result=result, probability=round(prob, 2), columns=columns)
        except Exception as e:
            return render_template('index.html', result=f"Error: {str(e)}", columns=columns)

    return render_template('index.html', result=None, columns=columns)

    
if __name__ == '__main__':
    app.run(debug=True)
