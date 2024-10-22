from flask import Flask, request, render_template
import pickle
import pandas as pd
import xgboost as xgb
from datetime import datetime
import calendar

# Load the trained model using pickle
model_path = 'Zomato_model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

# Updated encoding dictionary based on the dataset and model
encoding_dict = {
    'Road_traffic_density': {'Low': 0, 'Medium': 1, 'High': 2, 'Jam': 3},
    'Type_of_order': {'Snack': 0, 'Drinks': 1, 'Buffet': 2, 'Meal': 3},
    'Type_of_vehicle': {'Bike': 0, 'Scooter': 1, 'Car': 2},
    'Festival': {'No': 0, 'Yes': 1},
    'City': {'Urban': 0, 'Semi-Urban': 1, 'Metropolitan': 2},
    'Weather_conditions': {'Sunny': 0, 'Stormy': 1, 'Cloudy': 2, 'Windy': 3, 'Fog': 4, 'Sandstorms': 5}
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from the form
    features = {
        'Delivery_person_Age': int(request.form['Delivery_person_Age']),
        'Delivery_person_Ratings': float(request.form['Delivery_person_Ratings']),
        'Road_traffic_density': request.form['Road_traffic_density'],
        'Vehicle_condition': int(request.form['Vehicle_condition']),
        'Type_of_order': request.form['Type_of_order'],
        'Type_of_vehicle': request.form['Type_of_vehicle'],
        'multiple_deliveries': int(request.form['multiple_deliveries']),
        'Festival': request.form['Festival'],
        'City': request.form['City'],
        'distance': float(request.form['distance']),
        'Weather_conditions': request.form['Weather_conditions']
    }

    # Automatically calculate the date-related features
    current_date = datetime.now()
    features.update({
        'day': current_date.day,
        'month': current_date.month,
        'quarter': (current_date.month - 1) // 3 + 1,
        'year': current_date.year,
        'day_of_week': current_date.weekday(),
        'is_month_start': int(current_date.day == 1),
        'is_month_end': int(current_date.day == calendar.monthrange(current_date.year, current_date.month)[1]),
        'is_quarter_start': int(current_date.month in [1, 4, 7, 10] and current_date.day == 1),
        'is_quarter_end': int(current_date.month in [3, 6, 9, 12] and current_date.day == calendar.monthrange(current_date.year, current_date.month)[1]),
        'is_year_start': int(current_date.month == 1 and current_date.day == 1),
        'is_year_end': int(current_date.month == 12 and current_date.day == 31),
        'is_weekend': int(current_date.weekday() >= 5)
    })

    # Initialize the unused features with default values
    features['city_code'] = 0  # You can change this default value to what your model expects
    features['time_for_order_prepared'] = 0  # Initialize with a reasonable value, or use mean from training

    # Manually encode categorical features using the updated encoding_dict
    for col in encoding_dict:
        features[col] = encoding_dict[col][features[col]]

    # Convert features to a pandas DataFrame
    feature_df = pd.DataFrame([features])

    # Make prediction using the model
    prediction = model.predict(feature_df)
    output = round(prediction[0]*1.4, 2)

    # Render the prediction on the HTML page
    return render_template('index.html', prediction_text='Predicted Delivery Time: {} minutes'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
