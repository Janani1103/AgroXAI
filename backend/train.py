import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import shap
import joblib
import os

def load_or_generate_dataset():
    data_path = "C:/Users/Janani Nethma/Desktop/ElectraXAI/backend/crop_yield.csv"
    if os.path.exists(data_path):
        print(f"Loading real dataset from {data_path}...")
        df = pd.read_csv(data_path)
        # Filter for Rice if we are focusing on rice yield
        if 'Crop' in df.columns:
            df = df[df['Crop'] == 'Rice'].drop('Crop', axis=1)
        return df
    
    print("Dataset not found. Please ensure crop_yield.csv is in the backend folder.")
    # Fallback to generating something similar if file is missing (for safety)
    from backend.generate_data import generate_rice_dataset
    return generate_rice_dataset()

def preprocess_data(df):
    print("Step 3: Data Preprocessing...")
    df = df.dropna()
    
    # Target column name from the CSV
    target = 'Yield_tons_per_hectare'
    
    # Remove outliers
    Q1 = df[target].quantile(0.25)
    Q3 = df[target].quantile(0.75)
    IQR = Q3 - Q1
    df_clean = df[(df[target] >= Q1 - 1.5 * IQR) & (df[target] <= Q3 + 1.5 * IQR)]
    
    return df_clean

def train_model(df):
    df = preprocess_data(df)
    print("\nStep 4: Model Training (AI Core)...")
    target = 'Yield_tons_per_hectare'
    X = df.drop(target, axis=1)
    y = df[target]
    
    # Categorize features based on the CSV
    num_features = ["Rainfall_mm", "Temperature_Celsius", "Days_to_Harvest"]
    cat_features = ["Region", "Soil_Type", "Fertilizer_Used", "Irrigation_Used", "Weather_Condition"]
    
    # Ensure categorical/boolean columns are strings for OneHotEncoder
    for col in cat_features:
        if col in X.columns:
            X[col] = X[col].astype(str)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
        ])
    
    X_processed = preprocessor.fit_transform(X)
    
    # Get feature names for SHAP
    cat_encoder = preprocessor.named_transformers_['cat']
    cat_feature_names = cat_encoder.get_feature_names_out(cat_features)
    feature_names = num_features + list(cat_feature_names)
    
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
    
    # Train XGBoost
    print("Training XGBoost Regressor...")
    model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    print(f"Model R2 Score: {r2_score(y_test, preds):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, preds):.4f}")
    
    # Step 6: XAI Layer
    print("\nCalculating SHAP Importance...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_processed)
    
    global_importance = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': global_importance
    }).sort_values(by='importance', ascending=False)

    print("\nTop 5 Influencing Factors:")
    print(importance_df.head(5))

    # Save artifacts
    os.makedirs('models', exist_ok=True)
    joblib.dump(preprocessor, 'models/preprocessor.joblib')
    joblib.dump(model, 'models/xgboost_model.joblib')
    joblib.dump(feature_names, 'models/feature_names.joblib')
    joblib.dump(importance_df.to_dict(orient='records'), 'models/global_importance.joblib')
    
    print("\nArtifacts saved in models/ directory.")

if __name__ == "__main__":
    df = load_or_generate_dataset()
    train_model(df)
