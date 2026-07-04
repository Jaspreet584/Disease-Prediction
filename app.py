import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score

app = FastAPI(title="Disease Prediction API", version="1.0.0")

# Global Cache structures
MODELS: Dict[str, Dict] = {"heart": {}, "diabetes": {}, "breast_cancer": {}}
SCALERS: Dict[str, StandardScaler] = {}
METRICS: Dict[str, Dict] = {}

# Feature Schemas per disease
DISEASE_FEATURES = {
    "heart": [
        "Age", "Sex", "Chest_Pain_Type", "Resting_Blood_Pressure", "Cholesterol",
        "Fasting_Blood_Sugar", "Resting_ECG", "Max_Heart_Rate", "Exercise_Angina", "ST_Depression"
    ],
    "diabetes": [
        "Pregnancies", "Glucose", "Blood_Pressure", "Skin_Thickness", "Insulin",
        "BMI", "Diabetes_Pedigree_Function", "Age"
    ],
    "breast_cancer": [
        "Radius_Mean", "Texture_Mean", "Perimeter_Mean", "Area_Mean", "Smoothness_Mean",
        "Compactness_Mean", "Concavity_Mean", "Concave_Points_Mean", "Symmetry_Mean", "Fractal_Dimension_Mean"
    ]
}

# ----------------- Synthetic Data Generators -----------------

def generate_heart_data(n_samples: int = 800, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    age = np.random.randint(29, 80, size=n_samples)
    sex = np.random.choice([0, 1], p=[0.32, 0.68], size=n_samples) # 1 = male, 0 = female
    cp = np.random.choice([0, 1, 2, 3], p=[0.48, 0.16, 0.28, 0.08], size=n_samples) # pain type
    trestbps = np.clip(np.random.normal(loc=131, scale=17, size=n_samples), 90, 200).astype(int)
    chol = np.clip(np.random.normal(loc=246, scale=50, size=n_samples), 120, 450).astype(int)
    fbs = np.random.choice([0, 1], p=[0.85, 0.15], size=n_samples)
    restecg = np.random.choice([0, 1, 2], p=[0.49, 0.49, 0.02], size=n_samples)
    thalach = np.clip(180 - (age - 30) * 0.6 + np.random.normal(0, 12, size=n_samples), 70, 202).astype(int)
    exang = np.random.choice([0, 1], p=[0.67, 0.33], size=n_samples)
    oldpeak = np.clip(np.random.exponential(scale=1.0, size=n_samples), 0.0, 6.2).round(1)
    
    df = pd.DataFrame({
        "Age": age, "Sex": sex, "Chest_Pain_Type": cp, "Resting_Blood_Pressure": trestbps,
        "Cholesterol": chol, "Fasting_Blood_Sugar": fbs, "Resting_ECG": restecg,
        "Max_Heart_Rate": thalach, "Exercise_Angina": exang, "ST_Depression": oldpeak
    })
    
    # Calculate heart disease risk index
    risk = (
        (df["Age"] - 40) * 0.4 +
        df["Sex"] * 10.0 +
        (df["Chest_Pain_Type"] == 0).astype(int) * 12.0 + # asymptomatic typical angina is higher risk indicator
        (df["Resting_Blood_Pressure"] - 120) * 0.15 +
        (df["Cholesterol"] - 200) * 0.05 +
        df["Fasting_Blood_Sugar"] * 8.0 -
        (df["Max_Heart_Rate"] - 120) * 0.2 +
        df["Exercise_Angina"] * 15.0 +
        df["ST_Depression"] * 18.0
    )
    
    noise = np.random.normal(0, 12, size=n_samples)
    final_risk = risk + noise
    threshold = np.percentile(final_risk, 55) # ~45% positive rate
    df["Heart_Disease_Status"] = (final_risk >= threshold).astype(int)
    return df

def generate_diabetes_data(n_samples: int = 800, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    pregnancies = np.random.poisson(lam=3.8, size=n_samples)
    pregnancies = np.clip(pregnancies, 0, 17)
    glucose = np.clip(np.random.normal(loc=121, scale=31, size=n_samples), 45, 200).astype(int)
    bp = np.clip(np.random.normal(loc=69, scale=12, size=n_samples), 38, 122).astype(int)
    skin = np.clip(np.random.normal(loc=20, scale=15, size=n_samples), 0, 99).astype(int)
    insulin = np.clip(np.random.exponential(scale=80, size=n_samples), 0, 846).astype(int)
    bmi = np.clip(np.random.normal(loc=32, scale=7, size=n_samples), 16.0, 67.0).round(1)
    pedigree = np.clip(np.random.lognormal(mean=-0.8, sigma=0.5, size=n_samples), 0.08, 2.42).round(3)
    age = np.random.randint(21, 82, size=n_samples)
    
    df = pd.DataFrame({
        "Pregnancies": pregnancies, "Glucose": glucose, "Blood_Pressure": bp,
        "Skin_Thickness": skin, "Insulin": insulin, "BMI": bmi,
        "Diabetes_Pedigree_Function": pedigree, "Age": age
    })
    
    risk = (
        df["Pregnancies"] * 2.5 +
        (df["Glucose"] - 100) * 0.45 +
        (df["Blood_Pressure"] - 70) * 0.08 +
        df["BMI"] * 1.2 +
        df["Diabetes_Pedigree_Function"] * 22.0 +
        df["Age"] * 0.15 +
        (df["Insulin"] > 140).astype(int) * 5.0
    )
    
    noise = np.random.normal(0, 15, size=n_samples)
    final_risk = risk + noise
    threshold = np.percentile(final_risk, 65) # ~35% positive rate
    df["Diabetes_Status"] = (final_risk >= threshold).astype(int)
    return df

def generate_breast_cancer_data(n_samples: int = 800, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    radius = np.clip(np.random.normal(14.1, 3.5, size=n_samples), 6.9, 28.1).round(2)
    texture = np.clip(np.random.normal(19.3, 4.3, size=n_samples), 9.7, 39.2).round(2)
    perimeter = (radius * 6.28 + np.random.normal(0, 2, size=n_samples)).round(1)
    area = (3.14 * (radius ** 2) + np.random.normal(0, 10, size=n_samples)).round(1)
    smoothness = np.clip(np.random.normal(0.096, 0.014, size=n_samples), 0.05, 0.16).round(4)
    compactness = np.clip(np.random.beta(a=2, b=5, size=n_samples) * 0.35, 0.01, 0.34).round(4)
    concavity = np.clip(np.random.beta(a=1, b=4, size=n_samples) * 0.45, 0.0, 0.42).round(4)
    concave_points = np.clip(concavity * 0.48 + np.random.normal(0, 0.01, size=n_samples), 0.0, 0.20).round(4)
    symmetry = np.clip(np.random.normal(0.18, 0.027, size=n_samples), 0.1, 0.3).round(4)
    fractal = np.clip(np.random.normal(0.062, 0.007, size=n_samples), 0.05, 0.097).round(4)
    
    df = pd.DataFrame({
        "Radius_Mean": radius, "Texture_Mean": texture, "Perimeter_Mean": perimeter, "Area_Mean": area,
        "Smoothness_Mean": smoothness, "Compactness_Mean": compactness, "Concavity_Mean": concavity,
        "Concave_Points_Mean": concave_points, "Symmetry_Mean": symmetry, "Fractal_Dimension_Mean": fractal
    })
    
    risk = (
        (df["Radius_Mean"] - 14) * 2.8 +
        (df["Texture_Mean"] - 19) * 0.8 +
        df["Concavity_Mean"] * 45.0 +
        df["Concave_Points_Mean"] * 85.0 +
        df["Compactness_Mean"] * 20.0 +
        (df["Smoothness_Mean"] - 0.09) * 50.0
    )
    
    noise = np.random.normal(0, 4.5, size=n_samples)
    final_risk = risk + noise
    threshold = np.percentile(final_risk, 62) # ~38% malignant rate
    df["Cancer_Status"] = (final_risk >= threshold).astype(int)
    return df

# ----------------- Machine Learning Training -----------------

def train_disease_models(disease: str):
    global MODELS, SCALERS, METRICS
    
    # Load dataset
    if disease == "heart":
        df = generate_heart_data()
        target = "Heart_Disease_Status"
    elif disease == "diabetes":
        df = generate_diabetes_data()
        target = "Diabetes_Status"
    else:
        df = generate_breast_cancer_data()
        target = "Cancer_Status"
        
    features = DISEASE_FEATURES[disease]
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    # Fit Scaler for this disease (used for SVM and Logistic Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    SCALERS[disease] = scaler
    
    # Instantiate models
    disease_models = {
        "svm": SVC(probability=True, kernel="rbf", C=1.0, random_state=42),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    }
    
    METRICS[disease] = {}
    
    for name, clf in disease_models.items():
        # Train on scaled features for SVM and Logistic Regression, otherwise on raw features
        if name in ["svm", "logistic_regression"]:
            clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_test_scaled)
            y_prob = clf.predict_proba(X_test_scaled)[:, 1]
        else:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)[:, 1]
            
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        step = max(1, len(fpr) // 40)
        roc_points = [{"fpr": float(f), "tpr": float(t)} for f, t in zip(fpr[::step], tpr[::step])]
        if len(roc_points) == 0 or roc_points[-1]["fpr"] != 1.0 or roc_points[-1]["tpr"] != 1.0:
            roc_points.append({"fpr": 1.0, "tpr": 1.0})
            
        MODELS[disease][name] = clf
        METRICS[disease][name] = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "roc_auc": float(auc),
            "roc_curve": roc_points
        }

@app.on_event("startup")
def startup():
    # Pre-train models for all diseases on startup
    for disease in ["heart", "diabetes", "breast_cancer"]:
        train_disease_models(disease)

# ----------------- Prediction Request Schemas -----------------

class HeartInput(BaseModel):
    age: int = Field(..., ge=1, le=120)
    sex: int = Field(..., ge=0, le=1)
    chest_pain_type: int = Field(..., ge=0, le=3)
    resting_blood_pressure: int = Field(..., ge=60, le=240)
    cholesterol: int = Field(..., ge=80, le=600)
    fasting_blood_sugar: int = Field(..., ge=0, le=1)
    resting_ecg: int = Field(..., ge=0, le=2)
    max_heart_rate: int = Field(..., ge=50, le=240)
    exercise_angina: int = Field(..., ge=0, le=1)
    st_depression: float = Field(..., ge=0.0, le=10.0)
    model_name: str = Field("random_forest")

class DiabetesInput(BaseModel):
    pregnancies: int = Field(..., ge=0, le=20)
    glucose: int = Field(..., ge=0, le=300)
    blood_pressure: int = Field(..., ge=0, le=200)
    skin_thickness: int = Field(..., ge=0, le=100)
    insulin: int = Field(..., ge=0, le=1000)
    bmi: float = Field(..., ge=0.0, le=80.0)
    diabetes_pedigree_function: float = Field(..., ge=0.0, le=3.0)
    age: int = Field(..., ge=1, le=120)
    model_name: str = Field("random_forest")

class BreastCancerInput(BaseModel):
    radius_mean: float = Field(..., ge=1.0, le=40.0)
    texture_mean: float = Field(..., ge=1.0, le=50.0)
    perimeter_mean: float = Field(..., ge=10.0, le=300.0)
    area_mean: float = Field(..., ge=50.0, le=3000.0)
    smoothness_mean: float = Field(..., ge=0.01, le=0.3)
    compactness_mean: float = Field(..., ge=0.001, le=0.5)
    concavity_mean: float = Field(..., ge=0.0, le=0.6)
    concave_points_mean: float = Field(..., ge=0.0, le=0.3)
    symmetry_mean: float = Field(..., ge=0.05, le=0.5)
    fractal_dimension_mean: float = Field(..., ge=0.01, le=0.2)
    model_name: str = Field("random_forest")

# ----------------- API Route Mappings -----------------

@app.get("/api/performance/{disease}")
def get_performance(disease: str):
    if disease not in MODELS:
        raise HTTPException(status_code=400, detail="Invalid disease selection")
    if disease not in METRICS or not METRICS[disease]:
        train_disease_models(disease)
    return METRICS[disease]

@app.post("/api/predict/heart")
def predict_heart(data: HeartInput):
    disease = "heart"
    model_name = data.model_name
    if model_name not in MODELS[disease]:
        raise HTTPException(status_code=400, detail="Invalid model selection")
        
    features = [
        data.age, data.sex, data.chest_pain_type, data.resting_blood_pressure, data.cholesterol,
        data.fasting_blood_sugar, data.resting_ecg, data.max_heart_rate, data.exercise_angina, data.st_depression
    ]
    
    return run_inference(disease, model_name, features)

@app.post("/api/predict/diabetes")
def predict_diabetes(data: DiabetesInput):
    disease = "diabetes"
    model_name = data.model_name
    if model_name not in MODELS[disease]:
        raise HTTPException(status_code=400, detail="Invalid model selection")
        
    features = [
        data.pregnancies, data.glucose, data.blood_pressure, data.skin_thickness, data.insulin,
        data.bmi, data.diabetes_pedigree_function, data.age
    ]
    
    return run_inference(disease, model_name, features)

@app.post("/api/predict/breast_cancer")
def predict_breast_cancer(data: BreastCancerInput):
    disease = "breast_cancer"
    model_name = data.model_name
    if model_name not in MODELS[disease]:
        raise HTTPException(status_code=400, detail="Invalid model selection")
        
    features = [
        data.radius_mean, data.texture_mean, data.perimeter_mean, data.area_mean, data.smoothness_mean,
        data.compactness_mean, data.concavity_mean, data.concave_points_mean, data.symmetry_mean, data.fractal_dimension_mean
    ]
    
    return run_inference(disease, model_name, features)

def run_inference(disease: str, model_name: str, features_list: List):
    clf = MODELS[disease][model_name]
    X_df = pd.DataFrame([features_list], columns=DISEASE_FEATURES[disease])
    
    if model_name in ["svm", "logistic_regression"]:
        scaler = SCALERS[disease]
        X_scaled = scaler.transform(X_df)
        pred = int(clf.predict(X_scaled)[0])
        prob = float(clf.predict_proba(X_scaled)[0][1])
    else:
        pred = int(clf.predict(X_df)[0])
        prob = float(clf.predict_proba(X_df)[0][1])
        
    # Scale default risk mapping
    risk_percentage = prob * 100
    
    if risk_percentage < 20:
        severity = "Low"
    elif risk_percentage < 45:
        severity = "Moderate"
    elif risk_percentage < 75:
        severity = "High"
    else:
        severity = "Critical"
        
    return {
        "status": "Positive" if pred == 1 else "Negative",
        "probability": prob,
        "risk_level": severity
    }

# ----------------- Hosting Assets -----------------

static_path = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_path):
    os.makedirs(static_path)

app.mount("/static", StaticFiles(directory=static_path), name="static")

@app.get("/", response_class=HTMLResponse)
def get_index():
    index_file = os.path.join(static_path, "index.html")
    if not os.path.exists(index_file):
        return HTMLResponse("<h3>Disease Dashboard static/index.html is missing.</h3>", status_code=404)
    with open(index_file, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8001, reload=True)
