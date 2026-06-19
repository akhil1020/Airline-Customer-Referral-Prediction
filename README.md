# Airline Customer Review Prediction

This project is a Django-based machine learning web app that predicts whether an airline customer is likely to recommend an airline based on structured ratings and free-text review content.

It combines:

- numeric service ratings such as seat comfort, cabin service, and value for money
- categorical inputs such as airline, traveller type, and cabin class
- natural language processing on the written customer review using TF-IDF
- a trained logistic regression model for final recommendation prediction

## Overview

The application provides a simple browser form where a user can:

- choose an airline
- select traveller type and class of journey
- rate multiple service categories using star inputs
- write a short customer review
- receive a prediction showing whether the customer is likely to recommend the airline, along with model confidence

This makes the project a compact end-to-end example of integrating machine learning with a Django frontend.

## Features

- Django web interface for prediction
- Pretrained machine learning model loaded from disk
- TF-IDF vectorization for review text
- Logistic regression classifier for recommendation prediction
- Star-based rating UI for better user experience
- SQLite-backed Django project structure
- Notebook and dataset included for experimentation and model work

## Tech Stack

- Python 3.12+
- Django 6
- scikit-learn
- pandas
- numpy
- scipy
- joblib
- matplotlib

## How It Works

The prediction pipeline uses three kinds of input:

1. Numeric features
   Ratings such as overall experience, seat comfort, cabin service, food and beverages, entertainment, ground service, and value for money.

2. Categorical features
   Airline name, traveller type, and cabin class are converted into one-hot encoded features.

3. Text features
   The customer review is cleaned and transformed using a saved TF-IDF vectorizer.

These features are combined into a single feature vector and passed into a pretrained logistic regression model. The model returns:

- `1` for likely to recommend
- `0` for likely not to recommend

The UI also shows the associated confidence score from `predict_proba`.

## Project Structure

```text
airline_project/
|-- config/                         # Django project configuration
|-- predictor/                      # Main Django app
|   |-- static/predictor/           # CSS and JavaScript assets
|   |-- templates/predictor/        # HTML templates
|   |-- urls.py                     # App routes
|   `-- views.py                    # Prediction logic
|-- data/
|   `-- data.csv                    # Source dataset
|-- models/
|   |-- logistic_regression_model.pkl
|   `-- tfidf_vectorizer.pkl        # Saved ML artifacts
|-- notebooks/
|   `-- Airline.ipynb               # Model experimentation notebook
|-- Airline.ipynb                   # Notebook copy in project root
|-- manage.py
|-- requirements.txt
|-- pyproject.toml
`-- README.md
```

## Dataset

The dataset in `data/data.csv` contains airline review records with fields such as:

- airline
- customer review text
- traveller type
- cabin class
- service ratings
- recommendation label

The target appears to be whether the customer recommended the airline.

## Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd airline_project
```

### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

Using `requirements.txt`:

```bash
pip install -r requirements.txt
```

Or using `pyproject.toml` with `uv`:

```bash
uv sync
```

## Running the Project

Start the Django development server:

```bash
python manage.py runserver
```

Then open:

```text
http://127.0.0.1:8000/
```

## Usage

1. Open the home page in your browser.
2. Select the airline, traveller type, and cabin class.
3. Fill in the rating fields.
4. Enter a customer review.
5. Click `Predict Recommendation`.
6. View the prediction result and confidence score.

## Important Files

- `predictor/views.py` handles model loading, text cleaning, feature construction, and prediction
- `predictor/templates/predictor/index.html` contains the main review form
- `predictor/templates/predictor/result.html` displays the prediction output
- `models/logistic_regression_model.pkl` stores the trained classifier
- `models/tfidf_vectorizer.pkl` stores the text vectorizer

## Troubleshooting

### Model not loaded

If the app shows a model loading error, confirm these files exist:

- `models/logistic_regression_model.pkl`
- `models/tfidf_vectorizer.pkl`

### Dependency issues

Make sure your environment uses a compatible Python version and all required packages are installed.

### Static files not styling correctly

In development, confirm `django.contrib.staticfiles` is enabled and that you are running the app through Django using `runserver`.

## Future Improvements

- Add model training scripts and evaluation metrics to the main app workflow
- Improve form validation and error handling
- Store prediction history
- Add charts or analytics for review trends
- Deploy the app to a cloud platform
- Add automated tests for views and prediction logic

## License

This project is available for learning and personal development. Add a formal license if you plan to share or distribute it publicly.
