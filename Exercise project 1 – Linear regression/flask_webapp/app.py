from flask import Flask, render_template, request
import pickle
import numpy as np
import os

# flask app run
app = Flask(__name__)

# Load the saved model earlier here
model_path = os.path.join(os.path.dirname(__file__), "linear_model.pkl")
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# set to homepage
@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        try:
            # collect form data and convert
            age = float(request.form['age'])
            bmi = float(request.form['bmi'])
            children = int(request.form['children'])
            # categorical values to numerical
            smoker = 1 if request.form['smoker'] == 'yes' else 0
            sex = 1 if request.form['sex'] == 'male' else 0
            region_northeast = 1 if request.form['region'] == 'northeast' else 0
            region_northwest = 1 if request.form['region'] == 'northwest' else 0
            region_southeast = 1 if request.form['region'] == 'southeast' else 0
            region_southwest = 1 if request.form['region'] == 'southwest' else 0

            # input features matching as sometime it causes error
            input_features = np.array([[age, sex, bmi, children, smoker,
                                        region_northeast, region_northwest,
                                        region_southeast, region_southwest]])

            prediction = model.predict(input_features)[0]
            prediction = round(prediction, 2)

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template('index.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
