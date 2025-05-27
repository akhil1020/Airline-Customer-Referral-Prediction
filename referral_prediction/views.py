from django.shortcuts import render
import pickle
import numpy as np
import json
import os

# Load model and feature columns
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../Saved_model/model.pkl')
COLUMNS_PATH = os.path.join(os.path.dirname(__file__), '../Saved_model/columns.json')

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(COLUMNS_PATH, "r") as f:
        columns = json.load(f)
    x_cols = columns['data_columns']
except FileNotFoundError:
    model = None
    x_cols = []
    print("Error: Model or columns file not found.")

# Function for prediction
def predict_recommendation(airline, traveller_type, cabin, overall, seat_comfort, cabin_service, food_bev, entertainment, ground_service, value_for_money):
    try:
        if model is None:
            return "Model not loaded"

        X = np.zeros(len(x_cols))
        X[x_cols.index('overall')] = overall
        X[x_cols.index('seat_comfort')] = seat_comfort
        X[x_cols.index('cabin_service')] = cabin_service
        X[x_cols.index('food_bev')] = food_bev
        X[x_cols.index('entertainment')] = entertainment
        X[x_cols.index('ground_service')] = ground_service
        X[x_cols.index('value_for_money')] = value_for_money

        airline_col = 'airline_' + airline
        traveller_col = 'traveller_type_' + traveller_type
        cabin_col = 'cabin_' + cabin

        if airline_col in x_cols:
            X[x_cols.index(airline_col)] = 1
        if traveller_col in x_cols:
            X[x_cols.index(traveller_col)] = 1
        if cabin_col in x_cols:
            X[x_cols.index(cabin_col)] = 1

        prediction = model.predict([X])[0]
        return prediction
    except Exception as e:
        return f"Error: {e}"

# Index page
def index(request):
    return render(request, 'index.html')

# Prediction view
def predictor(request):
    if request.method == 'POST':
        airline = request.POST.get("airline", "")
        traveller_type = request.POST.get("traveller_type", "")
        cabin = request.POST.get("cabin", "")
        overall = float(request.POST.get("overall", 0))
        seat_comfort = float(request.POST.get("seat_comfort", 0))
        cabin_service = float(request.POST.get("cabin_service", 0))
        food_bev = float(request.POST.get("food_bev", 0))
        entertainment = float(request.POST.get("entertainment", 0))
        ground_service = float(request.POST.get("ground_service", 0))
        value_for_money = float(request.POST.get("value_for_money", 0))

        prediction = predict_recommendation(airline, traveller_type, cabin, overall, seat_comfort, cabin_service, food_bev, entertainment, ground_service, value_for_money)
        
        return render(request, 'result.html', {'result': prediction})
    return render(request, 'result.html', {'error': "Invalid request"})
