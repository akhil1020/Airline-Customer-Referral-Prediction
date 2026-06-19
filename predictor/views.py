from django.shortcuts import render
from django.conf import settings
import joblib
import re
import string
import numpy as np

try:
    model = joblib.load(settings.ML_MODEL_PATH)
    tfidf = joblib.load(settings.ML_TFIDF_PATH)
except FileNotFoundError as e:
    model = None
    tfidf = None
    print(f"Error loading model: {e}")

# Exact feature order the model was trained on:
# 7 numeric cols (explicit) + 85 one-hot cols + TF-IDF (minus 'overall' and 'entertainment'
# which were deduped because they appear in both the structured df and TF-IDF vocab)
NUMERIC_COLS = [
    'overall', 'seat_comfort', 'cabin_service',
    'food_bev', 'entertainment', 'ground_service', 'value_for_money',
]

# 80 airline cols (drop_first removed 'Adria Airways')
AIRLINE_COLS = [
    'airline_Aegean Airlines', 'airline_Aer Lingus',
    'airline_Aeroflot Russian Airlines', 'airline_Aeromexico', 'airline_Air Arabia',
    'airline_Air Canada', 'airline_Air Canada rouge', 'airline_Air China',
    'airline_Air Europa', 'airline_Air France', 'airline_Air India',
    'airline_Air New Zealand', 'airline_AirAsia', 'airline_Alaska Airlines',
    'airline_Alitalia', 'airline_American Airlines', 'airline_Asiana Airlines',
    'airline_Austrian Airlines', 'airline_Avianca', 'airline_Bangkok Airways',
    'airline_British Airways', 'airline_Brussels Airlines',
    'airline_Cathay Pacific Airways', 'airline_China Eastern Airlines',
    'airline_China Southern Airlines', 'airline_Copa Airlines',
    'airline_Delta Air Lines', 'airline_EVA Air', 'airline_Egyptair',
    'airline_Emirates', 'airline_Ethiopian Airlines', 'airline_Etihad Airways',
    'airline_Eurowings', 'airline_Finnair', 'airline_Frontier Airlines',
    'airline_Garuda Indonesia', 'airline_Germanwings', 'airline_Gulf Air',
    'airline_Iberia', 'airline_Icelandair', 'airline_IndiGo',
    'airline_Jetblue Airways', 'airline_KLM Royal Dutch Airlines',
    'airline_Korean Air', 'airline_Kuwait Airways', 'airline_LATAM Airlines',
    'airline_LOT Polish Airlines', 'airline_Lufthansa', 'airline_Norwegian',
    'airline_Pegasus Airlines', 'airline_Qantas Airways', 'airline_QantasLink',
    'airline_Qatar Airways', 'airline_Royal Air Maroc',
    'airline_Royal Jordanian Airlines', 'airline_Ryanair', 'airline_SAS Scandinavian',
    'airline_Saudi Arabian Airlines', 'airline_Singapore Airlines',
    'airline_South African Airways', 'airline_Southwest Airlines',
    'airline_Spirit Airlines', 'airline_Sunwing Airlines',
    'airline_Swiss Intl Air Lines', 'airline_TAP Portugal', 'airline_TAROM Romanian',
    'airline_Thai Airways', 'airline_Thai Smile Airways', 'airline_Tunisair',
    'airline_Turkish Airlines', 'airline_Ukraine International',
    'airline_United Airlines', 'airline_Virgin America', 'airline_Vueling Airlines',
    'airline_WOW air', 'airline_Wizz Air', 'airline_airBaltic', 'airline_easyJet',
    'airline_flydubai',
]

# 4 traveller cols (drop_first removed 'Business')
TRAVELLER_COLS = [
    'traveller_type_Couple Leisure', 'traveller_type_Family Leisure',
    'traveller_type_Solo Leisure', 'traveller_type_Unknown',
]

# 4 cabin cols (drop_first removed 'Business Class')
CABIN_COLS = [
    'cabin_Economy Class', 'cabin_First Class',
    'cabin_Premium Economy', 'cabin_Unknown',
]

# TF-IDF words that collide with structured cols are dropped by dedup:
# 'overall' and 'entertainment' exist in both numeric_cols and TF-IDF vocab.
# The dedup keeps the first occurrence (structured), so these TF-IDF positions are removed.
TFIDF_DEDUP_REMOVE = {'overall', 'entertainment'}


def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text


def build_feature_vector(airline, traveller_type, cabin, overall, seat_comfort,
                          cabin_service, food_bev, entertainment, ground_service,
                          value_for_money, customer_review):
    # 1. Numeric features
    numeric = np.array([
        overall, seat_comfort, cabin_service,
        food_bev, entertainment, ground_service, value_for_money,
    ], dtype=np.float32)

    # 2. One-hot: airline (0 = Adria Airways baseline)
    airline_vec = np.zeros(len(AIRLINE_COLS), dtype=np.float32)
    col = f'airline_{airline}'
    if col in AIRLINE_COLS:
        airline_vec[AIRLINE_COLS.index(col)] = 1.0

    # 3. One-hot: traveller type (0 = Business baseline)
    traveller_vec = np.zeros(len(TRAVELLER_COLS), dtype=np.float32)
    col = f'traveller_type_{traveller_type}'
    if col in TRAVELLER_COLS:
        traveller_vec[TRAVELLER_COLS.index(col)] = 1.0

    # 4. One-hot: cabin (0 = Business Class baseline)
    cabin_vec = np.zeros(len(CABIN_COLS), dtype=np.float32)
    col = f'cabin_{cabin}'
    if col in CABIN_COLS:
        cabin_vec[CABIN_COLS.index(col)] = 1.0

    # 5. TF-IDF — exclude the 2 words that were deduped out during training
    cleaned = clean_text(customer_review)
    tfidf_names = tfidf.get_feature_names_out()
    tfidf_full = tfidf.transform([cleaned]).toarray()[0].astype(np.float32)
    tfidf_vec = np.array(
        [v for name, v in zip(tfidf_names, tfidf_full) if name not in TFIDF_DEDUP_REMOVE],
        dtype=np.float32,
    )

    result = np.concatenate([numeric, airline_vec, traveller_vec, cabin_vec, tfidf_vec]).reshape(1, -1)
    return result


def index(request):
    return render(request, 'predictor/index.html')


def predictor(request):
    if request.method == 'POST':
        if model is None or tfidf is None:
            return render(request, 'predictor/result.html', {'error': 'Model not loaded. Check server logs.'})

        try:
            airline         = request.POST.get('airline', '')
            traveller_type  = request.POST.get('traveller_type', '')
            cabin           = request.POST.get('cabin', '')
            overall         = float(request.POST.get('overall', 0))
            seat_comfort    = float(request.POST.get('seat_comfort', 0))
            cabin_service   = float(request.POST.get('cabin_service', 0))
            food_bev        = float(request.POST.get('food_bev', 0))
            entertainment   = float(request.POST.get('entertainment', 0))
            ground_service  = float(request.POST.get('ground_service', 0))
            value_for_money = float(request.POST.get('value_for_money', 0))
            customer_review = request.POST.get('customer_review', '')

            X = build_feature_vector(
                airline, traveller_type, cabin,
                overall, seat_comfort, cabin_service,
                food_bev, entertainment, ground_service,
                value_for_money, customer_review,
            )

            prediction  = model.predict(X)[0]
            probability = model.predict_proba(X)[0][int(prediction)]

            return render(request, 'predictor/result.html', {
                'result': int(prediction),
                'probability': round(float(probability) * 100, 1),
            })

        except Exception as e:
            return render(request, 'predictor/result.html', {'error': str(e)})

    return render(request, 'predictor/result.html', {'error': 'Invalid request'})
