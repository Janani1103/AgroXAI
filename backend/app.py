from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import joblib
import os

app = Flask(__name__)
CORS(app)

# Load artifacts globally
model = None
preprocessor = None
feature_names = None
explainer = None

def load_artifacts():
    global model, preprocessor, feature_names, explainer
    model_path = 'models/xgboost_model.joblib'
    prep_path = 'models/preprocessor.joblib'
    feat_path = 'models/feature_names.joblib'
    
    if os.path.exists(model_path) and os.path.exists(prep_path):
        model = joblib.load(model_path)
        preprocessor = joblib.load(prep_path)
        feature_names = joblib.load(feat_path)
        explainer = shap.TreeExplainer(model)
        print("Models loaded successfully.")
    else:
        print("Model files not found. Please run train.py first.")

load_artifacts()

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Parse inputs matching crop_yield.csv schema
        input_data = {
            'Rainfall_mm': float(data.get('rainfall', 0)),
            'Temperature_Celsius': float(data.get('temperature', 0)),
            'Days_to_Harvest': float(data.get('daysToHarvest', 120)),
            'Region': data.get('region', 'South'),
            'Soil_Type': data.get('soilType', 'Clay'),
            'Fertilizer_Used': str(data.get('fertilizerUsed', 'True')),
            'Irrigation_Used': str(data.get('irrigationUsed', 'True')),
            'Weather_Condition': data.get('weatherCondition', 'Sunny')
        }
        
        df = pd.DataFrame([input_data])
        
        # Preprocess
        X_processed = preprocessor.transform(df)
        
        # Predict
        prediction = model.predict(X_processed)[0]
        
        # SHAP explanation
        shap_values = explainer.shap_values(X_processed)[0]
        
        # Map SHAP values back to clean feature names
        shap_dict = {}
        for i, val in enumerate(shap_values):
            feat = feature_names[i]
            # Clean feature names (e.g., cat__Region_South -> Region)
            if feat.startswith('cat__'):
                clean_name = feat.split('__')[1].split('_')[0]
                # Combine impacts for the same category group
                shap_dict[clean_name] = shap_dict.get(clean_name, 0) + val
            elif feat.startswith('num__'):
                clean_name = feat.replace('num__', '').replace('_', ' ')
                shap_dict[clean_name] = val
        
        # Sort by absolute impact
        sorted_shap = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        top_shap = [{"feature": k, "impact": float(v)} for k, v in sorted_shap]
        
        # Base value (expected value)
        base_value = float(explainer.expected_value)
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[0]
            
        return jsonify({
            'success': True,
            'prediction': float(prediction),
            'base_value': base_value,
            'shap_values': top_shap
        })

    except Exception as e:
        print("Error:", str(e))
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
