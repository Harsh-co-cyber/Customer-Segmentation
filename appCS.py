from flask import Flask, render_template, request, redirect, flash
import joblib
import pandas as pd

appCS = Flask(__name__)
appCS.secret_key = 'your_secret_key'  # Replace with a strong key

# Load the trained model and scaler
model = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')

# Define cluster descriptions
cluster_descriptions = {
    0: "Cluster 0: High-value customers who make frequent purchases but with lower recency. They are the most loyal and engaged.",
    1: "Cluster 1: Customers with very high monetary values and frequency, but low recency. These are highly valuable customers with recent transactions.",
    2: "Cluster 2: Moderate-value customers with average frequency and recency. They purchase regularly but with moderate spending.",
    3: "Cluster 3: Low-frequency and high recency customers. These are less engaged and have not purchased recently."
}

# Home Page
@appCS.route('/')
def home():
    return render_template('home.html')

# About Us Page
@appCS.route('/about')
def about():
    return render_template('about.html')

# Predict Route
@appCS.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Retrieve and convert form data
            recency = int(request.form['recency'])
            monetary = float(request.form['monetary'])
            frequency = int(request.form['frequency'])

            # Prepare the data for prediction
            data = pd.DataFrame({
                'Recency': [recency],
                'Monetary': [monetary],
                'Frequency': [frequency]
            })

            # Standardize the data
            scaled_data = scaler.transform(data)

            # Predict using the loaded model
            prediction = model.predict(scaled_data)[0]

            # Get cluster description
            description = cluster_descriptions.get(prediction, "No description available for this cluster.")

            return render_template('result1.html', cluster=prediction, description=description)

        except ValueError as e:
            return render_template('error.html', message=f"Input Error: {e}. Please ensure all inputs are valid.")
        except Exception as e:
            return render_template('error.html', message=f"Unexpected error: {e}. Please try again later.")
    
    # Render the prediction form
    return render_template('predict.html')  # Create this template for input

if __name__ == '__main__':
    appCS.run(debug=True)
