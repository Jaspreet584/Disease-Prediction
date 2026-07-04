# PathoAI // Disease Prediction from Medical Data

PathoAI is an interactive medical risk assessment web application designed to predict the likelihood of disease from patient clinical parameters. It trains multiple machine learning classifiers (SVM, Logistic Regression, Random Forest, and Gradient Boosting) across three distinct health datasets and offers real-time risk diagnostic estimations.

## Key Features

- **Multi-Disease Targets**: Evaluate diagnostic risks for:
  - **Heart Disease**: Analyze physiological symptoms (chest pain types, exercise-induced angina, ST depression).
  - **Diabetes Mellitus**: Track glycemic and metabolic variables (glucose levels, insulin, BMI).
  - **Breast Cancer**: Predict cell malignancy based on nuclear dimensions (radius, perimeter, concavity).
- **Machine Learning Classifiers**: Automatically trains and evaluates 4 models:
  - **Support Vector Machine (SVM)** (utilizing RBF kernels and feature standardization)
  - **Logistic Regression** (with feature standardization)
  - **Random Forest Classifier**
  - **Gradient Boosting Classifier**
- **Dynamic Frontend Dashboard**: The range sliders and output labels dynamically adapt to the selected target disease.
- **Automated Performance Comparisons**: Computes and compares accuracy, precision, recall, f1-score, and ROC-AUC side-by-side using Chart.js bar and scatter charts.

## Technology Stack

- **Backend**: Python 3, FastAPI, scikit-learn, pandas, numpy, uvicorn
- **Frontend**: HTML5, Vanilla CSS3, JavaScript (ES6+), Chart.js, FontAwesome Icons

## Project Structure

```text
├── app.py              # FastAPI backend, data generator pipelines, and models
├── static/
    ├── index.html      # Multi-disease dashboard layout
    ├── styles.css      # Dark-theme CSS system with teal highlights
    └── app.js          # Dynamic slider forms, APIs, and Chart.js integrations
```

## Getting Started

### Prerequisites

Ensure you have Python 3 installed. Install the required dependencies:

```bash
pip install pandas scikit-learn numpy fastapi uvicorn pydantic python-multipart
```

### Running the Application

1. Open your terminal and navigate to the project directory:
   ```bash
   cd "Task 2 - Disease Prediction"
   ```
2. Launch the local FastAPI server:
   ```bash
   python app.py
   ```
3. Open your browser and navigate to:
   [http://localhost:8000](http://localhost:8000)
