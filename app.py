from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the trained model
with open('electricity_model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input from the form
        consumption = float(request.form['consumption'])
        hours = float(request.form['hours'])
        appliances = float(request.form['appliances'])

        # Create a feature array
        features = np.array([[consumption, hours, appliances]])

        # Predict using the loaded model
        prediction = model.predict(features)

        # Return the prediction result
        return render_template('index.html', prediction_text=f"Estimated Price: ${prediction[0]:.2f}")

    except ValueError as e:
        return render_template('index.html', prediction_text="Invalid input, please try again.")

if __name__ == '__main__':
    app.run(debug=True)
