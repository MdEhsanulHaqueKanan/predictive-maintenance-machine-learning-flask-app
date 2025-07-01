import os
from flask import Flask, request, render_template
import logging
import joblib
import pandas as pd
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# --- BEST PRACTICE: Use an absolute path to load the model ---
# This makes the app more robust
basedir = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.join(basedir, 'saved_models', 'best_failure_predictor.joblib')

# Load the trained model pipeline
try:
    model = joblib.load(model_path)
    app.logger.info(f"Model loaded successfully from: {model_path}")
except Exception as e:
    app.logger.error(f"Error loading model from {model_path}: {e}")
    model = None


@app.route('/')
def home():
    """Renders the home page with the input form."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Receives form data, performs feature engineering, makes a prediction, and displays the result."""
    if model is None:
        app.logger.error("Prediction attempted but model is not loaded.")
        return "Model not loaded. Please check the logs.", 500

    try:
        # 1. Collect form data from the request
        required_fields = {
            'type': str, 'air_temp': float, 'process_temp': float,
            'rotational_speed': float, 'torque': float, 'vibration': float,
            'op_hours': float
        }
        form_data_raw = {field: request.form.get(field) for field in required_fields}

        # Check for missing fields
        if any(value is None for value in form_data_raw.values()):
            return "Error: Missing form data. Please fill out all fields.", 400

        # Create a dictionary with the correct column names and types
        form_data = {
            'Type': form_data_raw['type'],
            'Air temperature [K]': float(form_data_raw['air_temp']),
            'Process temperature [K]': float(form_data_raw['process_temp']),
            'Rotational speed [rpm]': float(form_data_raw['rotational_speed']),
            'Torque [Nm]': float(form_data_raw['torque']),
            'Vibration Levels': float(form_data_raw['vibration']),
            'Operational Hours': float(form_data_raw['op_hours']),
        }

        # 2. Convert the collected data into a pandas DataFrame
        input_df = pd.DataFrame([form_data])

        # --- CRITICAL FIX: Perform the EXACT same feature engineering as in the notebook ---
        # 3. Create the engineered features that the model was trained on
        
        # Create Temperature Difference feature
        input_df['Temp_diff_K'] = input_df['Process temperature [K]'] - input_df['Air temperature [K]']

        # Create Mechanical Power feature
        input_df['Power_W'] = np.round((input_df['Torque [Nm]'] * input_df['Rotational speed [rpm]'] * 2 * np.pi) / 60, 4)
        
        # Now, input_df has all the columns the model expects.

        # 4. Make a prediction and get probabilities
        # The model's internal pipeline will handle selecting the right columns and scaling them.
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)
        
        # Get the confidence for the specific predicted class
        pred_class_index = np.where(model.classes_ == prediction)[0][0]
        confidence = round(probabilities[0, pred_class_index] * 100, 2)

        # 5. Prepare the output message for the user
        result = {
            'prediction': prediction,
            'confidence': confidence,
            'risk_factor': None, # Default value
        }
        
        # Add intelligent recommendations based on the prediction
        if prediction == 'No Failure':
            result['status_text'] = 'Status: Normal Operation'
            result['emoji'] = '✅'
            result['recommendation'] = 'Continue routine monitoring.'
        else:
            result['status_text'] = 'Alert: Failure Predicted!'
            result['emoji'] = '⚠️'
            # Simple rule-based logic to provide a more specific recommendation
            if 'Heat' in prediction and form_data['Process temperature [K]'] > 303:
                result['risk_factor'] = f"The Process Temperature ({form_data['Process temperature [K]']} K) is unusually high."
                result['recommendation'] = "Schedule immediate inspection of the cooling system."
            elif 'Power' in prediction:
                 result['risk_factor'] = f"The combination of Torque ({form_data['Torque [Nm]']} Nm) and Speed ({form_data['Rotational speed [rpm]']} rpm) suggests a power transmission issue."
                 result['recommendation'] = "Inspect the motor and power supply."
            else:
                 result['risk_factor'] = f"High Vibration Levels ({form_data['Vibration Levels']}) detected."
                 result['recommendation'] = "Inspect the machine for mechanical wear or imbalance."

        # 6. Render the result page with all the prepared information
        return render_template('result.html', result=result)

    except Exception as e:
        # Handle any other unexpected errors gracefully
        app.logger.error(f"An error occurred during prediction: {e}", exc_info=True)
        return f"An error occurred: {e}", 500


if __name__ == '__main__':
    # Runs the app on a local development server
    # debug=True will automatically reload the server when you save changes
    # For production, use a proper WSGI server like Gunicorn or uWSGI
    # and set debug=False
    app.run(debug=os.environ.get('FLASK_DEBUG', 'True').lower() == 'true')