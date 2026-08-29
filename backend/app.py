# Import necessary libraries
import numpy as np
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API

# Initialize the Flask application
superkart_predictor_api = Flask("SuperKart Sales Predictor")

# Load the trained machine learning model
model = joblib.load("superkart_model.joblib")

# Define a route for the home page (GET request)
@superkart_predictor_api.get('/')
def home():
    """
    Handles GET requests to the root URL ('/').
    Returns a simple welcome message.
    """
    return "Welcome to the SuperKart Sales Prediction API!"

# Define an endpoint for single prediction (POST request)
@superkart_predictor_api.post('/v1/predict')
def predict_sales():
    """
    Handles POST requests to '/v1/predict'.
    Expects a JSON payload containing product/store details.
    Returns the predicted sales value as JSON.
    """
    # Get the JSON data from the request body
    product_data = request.get_json()

    # Convert the extracted data into a Pandas DataFrame
    input_data = pd.DataFrame([product_data])

    # Make prediction
    predicted_sales = model.predict(input_data)[0]

    # Convert to Python float for JSON serialization
    predicted_sales = round(float(predicted_sales), 2)

    # Return the prediction
    return jsonify({'Predicted Sales': predicted_sales})

# Define an endpoint for batch prediction (POST request)
@superkart_predictor_api.post('/v1/predictbatch')
def predict_sales_batch():
    """
    Handles POST requests to '/v1/predictbatch'.
    Expects a CSV file containing multiple product/store records.
    Returns predicted sales values as a dictionary keyed by row index.
    """
    # Get the uploaded CSV file from the request
    file = request.files['file']

    # Read the CSV file into a Pandas DataFrame
    input_data = pd.read_csv(file)

    # Make predictions for all rows
    predicted_sales = model.predict(input_data).tolist()

    # Convert predictions to floats and round
    predicted_sales = [round(float(p), 2) for p in predicted_sales]

    # Create a dictionary of predictions keyed by row index
    output_dict = {i: val for i, val in enumerate(predicted_sales)}

    # Return the predictions dictionary as JSON
    return jsonify(output_dict)

# Run the Flask application in debug mode if executed directly
if __name__ == '__main__':
    superkart_predictor_api.run(debug=True)
