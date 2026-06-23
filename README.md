# climate-prediction
Climate Risk Prediction System using RNN and LSTM
Project Overview

The Climate Risk Prediction System is an AI-powered web application developed to predict weather conditions using environmental and climate parameters. The system utilizes deep learning techniques, specifically Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) networks, to analyze climate data and generate weather predictions.

The application allows users to input climate parameters such as temperature, humidity, atmospheric pressure, wind speed, precipitation, cloud cover, and UV index. Based on these inputs, the system predicts the most likely weather condition and compares the performance of both RNN and LSTM models.

Features
Climate risk prediction using deep learning.
Weather condition classification.
RNN and LSTM model implementation.
Automatic model comparison.
Confidence score generation.
Real-time prediction through Flask API.
Interactive web-based user interface.
Model training and status monitoring.
Best-performing model selection.
Technologies Used
Python
TensorFlow
Keras
NumPy
Pandas
Scikit-Learn
Flask
HTML
CSS
JavaScript
Dataset

The project uses a climate dataset containing environmental parameters such as:

Temperature (°C)
Humidity (%)
Pressure (hPa)
Wind Speed (m/s)
Precipitation (mm)
Cloud Cover (%)
UV Index

Target Variable:

Weather Class
Project Workflow
Data Collection
Data Preprocessing
Feature Selection
Label Encoding
Feature Scaling
Train-Test Split
RNN Model Training
LSTM Model Training
Model Evaluation
Prediction Generation
Model Comparison
Web Deployment
Installation

Clone the repository:

git clone https://github.com/your-username/climate-risk-prediction.git

Navigate to the project directory:

cd climate-risk-prediction

Install dependencies:

pip install -r requirements.txt

Run the application:

python app.py
Results

The system successfully predicts weather conditions using climate parameters and compares the performance of RNN and LSTM models. The application provides confidence scores, model accuracies, and identifies the best-performing model for prediction.

Future Enhancements
Integration of real-time weather APIs.
Advanced deep learning architectures.
Cloud deployment.
Mobile application support.
Climate risk alert generation.
Interactive data visualization dashboards.
