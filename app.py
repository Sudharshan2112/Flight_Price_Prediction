"""
=============================================================
  SKYFARE — FLIGHT PRICE PREDICTION
  Backend API  : Flask REST Server
  Connects to  : skyfare_v2.html  (frontend)
  Models used  : RF + HGB + ET Weighted Ensemble
                 + RandomForest Seat Class Classifier
  Run          : python app.py
  API endpoint : POST /predict
=============================================================
"""

import os
import json
import pickle
import warnings
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────────────────
app = Flask(__name__, static_folder='.')
CORS(app)  # Allow requests from the HTML frontend

# ─────────────────────────────────────────────────────────
# LOAD MODEL ARTIFACTS
# (Run flight_model_v2.py first to generate these)
# ─────────────────────────────────────────────────────────
MODELS_DIR = './models'

def load_models():
    """Load all trained model artifacts from disk."""
    artifacts = {}
    try:
        with open(os.path.join(MODELS_DIR, 'rf_model.pkl'),  'rb') as f: artifacts['rf']  = pickle.load(f)
        with open(os.path.join(MODELS_DIR, 'hgb_model.pkl'), 'rb') as f: artifacts['hgb'] = pickle.load(f)
        with open(os.path.join(MODELS_DIR, 'et_model.pkl'),  'rb') as f: artifacts['et']  = pickle.load(f)
        with open(os.path.join(MODELS_DIR, 'clf_model.pkl'), 'rb') as f: artifacts['clf'] = pickle.load(f)
        with open(os.path.join(MODELS_DIR, 'encoders.pkl'),  'rb') as f: artifacts['enc'] = pickle.load(f)
        with open(os.path.join(MODELS_DIR, 'model_info.json'), 'r') as f: artifacts['info'] = json.load(f)
        print("  ✓ All model artifacts loaded successfully")
    except FileNotFoundError as e:
        print(f"  ✗ Model files not found: {e}")
        print("  → Please run flight_model_v2.py first to train and save the models.")
        artifacts = None
    return artifacts

artifacts = load_models()

# ─────────────────────────────────────────────────────────
# ENCODING MAPS  (must match flight_model_v2.py exactly)
# ─────────────────────────────────────────────────────────
AIRLINE_ENC = {
    'Air Asia': 0, 'Air India': 1, 'GoAir': 2, 'IndiGo': 3,
    'Jet Airways': 4, 'Jet Airways Business': 5, 'Multiple carriers': 6,
    'Multiple carriers Premium economy': 7, 'SpiceJet': 8, 'Trujet': 9,
    'Vistara': 10, 'Vistara Premium economy': 11
}
SOURCE_ENC = {
    'Banglore': 0, 'Chennai': 1, 'Delhi': 2, 'Kolkata': 3, 'Mumbai': 4
}
DEST_ENC = {
    'Banglore': 0, 'Cochin': 1, 'Delhi': 2,
    'Hyderabad': 3, 'Kolkata': 4, 'New Delhi': 5
}
AIRLINE_TIERS = {
    'IndiGo': 1, 'SpiceJet': 1, 'GoAir': 1, 'Air Asia': 1, 'Trujet': 1,
    'Jet Airways': 2, 'Air India': 2, 'Multiple carriers': 2,
    'Vistara': 3,
    'Vistara Premium economy': 4, 'Multiple carriers Premium economy': 4,
    'Jet Airways Business': 5
}
STOPS_MAP = {
    'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4
}

# Feature column order — must exactly match training
FEATURE_COLS = [
    'Airline', 'Source', 'Destination',
    'Journey_Day', 'Journey_Month', 'Journey_Weekday',
    'Is_Weekend', 'Is_Month_Start', 'Is_Month_End', 'Is_Peak_Month',
    'Dep_Hour', 'Dep_Min', 'Dep_Slot',
    'Arr_Hour', 'Arr_Min',
    'Duration_mins', 'Duration_hrs',
    'Stops', 'Is_Direct', 'Num_Routes',
    'Airline_Tier', 'Info_Code', 'Speed_proxy'
]

# ─────────────────────────────────────────────────────────
# HELPER — TIME SLOT
# ─────────────────────────────────────────────────────────
def get_time_slot(hour: int) -> int:
    if 4 <= hour < 8:   return 0   # Early Morning
    if 8 <= hour < 12:  return 1   # Morning
    if 12 <= hour < 17: return 2   # Afternoon
    if 17 <= hour < 21: return 3   # Evening
    return 4                        # Night

# ─────────────────────────────────────────────────────────
# HELPER — BUILD FEATURE VECTOR FROM REQUEST
# ─────────────────────────────────────────────────────────
def build_features(data: dict) -> np.ndarray:
    """
    Convert raw form fields from the HTML frontend into
    the exact feature vector the models were trained on.
    """
    airline      = data.get('airline', '')
    source       = data.get('source', '')
    destination  = data.get('destination', '')
    journey_date = data.get('journey_date', '')       # YYYY-MM-DD
    dep_time     = data.get('dep_time', '00:00')      # HH:MM
    arr_time     = data.get('arr_time', '00:00')      # HH:MM
    duration_hrs = float(data.get('duration_hrs', 0)) # decimal hours
    stops        = int(data.get('stops', 0))
    info_code    = int(data.get('info_code', 0))

    # ── Date
    from datetime import datetime
    dt = datetime.strptime(journey_date, '%Y-%m-%d')
    journey_day     = dt.day
    journey_month   = dt.month
    journey_weekday = dt.weekday()           # 0=Mon … 6=Sun
    is_weekend      = 1 if journey_weekday >= 5 else 0
    is_month_start  = 1 if journey_day <= 5 else 0
    is_month_end    = 1 if journey_day >= 25 else 0
    is_peak_month   = 1 if journey_month in [5, 6, 11, 12] else 0

    # ── Times
    dep_h, dep_m = map(int, dep_time.split(':'))
    arr_h, arr_m = map(int, arr_time.split(':')[:2])  # strip any AM/PM suffix
    dep_slot = get_time_slot(dep_h)

    # ── Duration
    if duration_hrs and duration_hrs > 0:
        dur_mins = duration_hrs * 60
    else:
        # derive from dep/arr times
        dur_mins = (arr_h * 60 + arr_m) - (dep_h * 60 + dep_m)
        if dur_mins <= 0:
            dur_mins += 1440   # overnight flight
    dur_hrs = dur_mins / 60.0

    # ── Derived
    is_direct   = 1 if stops == 0 else 0
    num_routes  = stops + 1
    airline_tier = AIRLINE_TIERS.get(airline, 2)
    speed_proxy  = dur_mins / (stops + 1)

    # ── Encoded categoricals
    airline_enc = AIRLINE_ENC.get(airline, 3)   # default: IndiGo
    source_enc  = SOURCE_ENC.get(source, 2)     # default: Delhi
    dest_enc    = DEST_ENC.get(destination, 1)  # default: Cochin

    feature_vector = [
        airline_enc, source_enc, dest_enc,
        journey_day, journey_month, journey_weekday,
        is_weekend, is_month_start, is_month_end, is_peak_month,
        dep_h, dep_m, dep_slot,
        arr_h, arr_m,
        dur_mins, dur_hrs,
        stops, is_direct, num_routes,
        airline_tier, info_code, speed_proxy
    ]

    return np.array(feature_vector).reshape(1, -1)

# ─────────────────────────────────────────────────────────
# ROUTE — Serve the frontend HTML
# ─────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('.', 'skyfare_v2.html')

# ─────────────────────────────────────────────────────────
# ROUTE — Health check
# ─────────────────────────────────────────────────────────
@app.route('/health')
def health():
    model_status = 'loaded' if artifacts else 'not loaded — run flight_model_v2.py first'
    return jsonify({'status': 'ok', 'models': model_status})

# ─────────────────────────────────────────────────────────
# ROUTE — Model metadata (for the frontend info panel)
# ─────────────────────────────────────────────────────────
@app.route('/model-info')
def model_info():
    if not artifacts:
        return jsonify({'error': 'Models not loaded'}), 503
    return jsonify(artifacts['info'])

# ─────────────────────────────────────────────────────────
# ROUTE — MAIN PREDICTION ENDPOINT
# POST /predict
# Body (JSON):
#   airline        : str   e.g. "IndiGo"
#   source         : str   e.g. "Delhi"
#   destination    : str   e.g. "Cochin"
#   journey_date   : str   e.g. "2025-06-15"  (YYYY-MM-DD)
#   dep_time       : str   e.g. "08:30"
#   arr_time       : str   e.g. "11:45"
#   duration_hrs   : float e.g. 3.25  (0 = auto-calculate)
#   stops          : int   e.g. 1
#   info_code      : int   0-8
#
# Returns (JSON):
#   predicted_price : int    e.g. 6420
#   price_range_low : int    e.g. 5743
#   price_range_high: int    e.g. 7097
#   seat_class      : str    "Economy" | "Premium Economy" | "Business"
#   confidence      : int    65–95
#   model_blend     : str    blend description
# ─────────────────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    if not artifacts:
        return jsonify({'error': 'Models not loaded. Run flight_model_v2.py first.'}), 503

    data = request.get_json(force=True)
    if not data:
        return jsonify({'error': 'No JSON body received'}), 400

    # Validate required fields
    required = ['airline', 'source', 'destination', 'journey_date', 'dep_time', 'arr_time']
    missing = [f for f in required if not data.get(f)]
    if missing:
        return jsonify({'error': f'Missing fields: {", ".join(missing)}'}), 400

    try:
        X = build_features(data)

        rf   = artifacts['rf']
        hgb  = artifacts['hgb']
        et   = artifacts['et']
        clf  = artifacts['clf']
        enc  = artifacts['enc']
        seat_le = enc['Seat_Class']

        # Weighted ensemble: RF 25% + HGB 45% + ET 30%
        price_rf  = float(rf.predict(X)[0])
        price_hgb = float(hgb.predict(X)[0])
        price_et  = float(et.predict(X)[0])
        price_ens = 0.25 * price_rf + 0.45 * price_hgb + 0.30 * price_et
        price_ens = max(1500, round(price_ens))

        # Seat class from classifier
        cls_idx   = clf.predict(X)[0]
        seat_class = seat_le.inverse_transform([cls_idx])[0]

        # Price confidence band (based on model CV MAE)
        info = artifacts['info']
        mae  = info.get('cv_mae', 677)
        price_low  = max(1000, round(price_ens - mae))
        price_high = round(price_ens + mae)

        # Confidence score (mirrors frontend logic, driven by actual CV R²)
        base_conf  = round(info.get('cv_r2', 0.82) * 100)
        stops      = int(data.get('stops', 0))
        airline_tier = AIRLINE_TIERS.get(data.get('airline', ''), 2)
        dur_given  = float(data.get('duration_hrs', 0)) > 0
        confidence = min(95, max(65,
            base_conf
            + (5 if stops == 0 else -2)
            + (4 if airline_tier >= 3 else 0)
            + (3 if dur_given else 0)
        ))

        return jsonify({
            'predicted_price' : price_ens,
            'price_range_low' : price_low,
            'price_range_high': price_high,
            'seat_class'      : seat_class,
            'confidence'      : confidence,
            'model_blend'     : info.get('blend', 'RF(25%) + HGB(45%) + ET(30%)'),
            'cv_r2'           : info.get('cv_r2', None),
            'cv_mae'          : info.get('cv_mae', None),
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "="*60)
    print("  SKYFARE — Flight Price Prediction API")
    print("="*60)
    print("  Frontend : http://localhost:5000/")
    print("  API      : http://localhost:5000/predict  [POST]")
    print("  Health   : http://localhost:5000/health")
    print("="*60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
