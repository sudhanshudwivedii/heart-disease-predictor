❤️ Heart Disease Prediction System

An end-to-end machine learning web app that predicts the risk of heart disease from a patient's clinical parameters. Built with scikit-learn and deployed live on Streamlit Cloud.

🔗 **Live App:** https://heart-disease-predictor-miykw9oi97ldcmhhndzv3z.streamlit.app/

---

## 📌 Overview

This project takes clinical inputs (age, cholesterol, chest pain type, resting ECG, and more) and predicts whether a patient is likely to have heart disease, along with a probability score. It covers the full machine learning lifecycle — from raw data cleaning to a live, interactive web application.

The goal was not just to train a model, but to build something a real user can open in their browser and interact with.

---

## ✨ Features

- Interactive web interface with sliders and dropdowns for patient inputs
- Real-time prediction with a probability score
- Clear risk indication (High Risk / Low Risk)
- Trained on 918 patient records with 11 clinical features
- Deployed publicly and accessible from any device

---

## 📊 Model Performance

Six classification algorithms were trained and compared using 5-fold cross-validation. **Random Forest** was selected as the final model because it had the best recall and F1-score — which matter most for a medical use case, where missing a true positive (a patient who actually has the disease) is the most costly error.

| Model          | Accuracy | Recall | Precision | F1    | CV Avg |
|----------------|----------|--------|-----------|-------|--------|
| **Random Forest** | **0.875** | **0.879** | 0.904 | **0.891** | 0.852 |
| SVM            | 0.837    | 0.832  | 0.881     | 0.856 | 0.860  |
| Logistic Regression | 0.848 | 0.822 | 0.907   | 0.863 | 0.842  |
| KNN            | 0.832    | 0.804  | 0.896     | 0.847 | 0.843  |
| Decision Tree  | 0.826    | 0.785  | 0.903     | 0.840 | 0.807  |
| Naive Bayes    | 0.799    | 0.757  | 0.880     | 0.814 | 0.857  |

---

## 🛠️ Tech Stack

- **Python**
- **scikit-learn** — model training and evaluation
- **pandas / numpy** — data processing
- **Streamlit** — web app and deployment
- **joblib** — model persistence

---

## 🔬 Data Pipeline

1. **Data cleaning** — Identified disguised missing values encoded as zeros in the `Cholesterol` column (physiologically impossible), converted them to NaN, and imputed with the median.
2. **Encoding** — Categorical features (Sex, Chest Pain Type, Resting ECG, Exercise Angina, ST Slope) were label-encoded.
3. **Feature scaling** — Applied `StandardScaler` after splitting the data (fit on train only) to prevent data leakage.
4. **Stratified train-test split** — 80/20 split preserving the class ratio.
5. **Model comparison** — Six algorithms evaluated with 5-fold cross-validation.
6. **Deployment** — Best model saved with joblib and served via Streamlit.

---

## 📁 Project Structure

```
heart-disease-predictor/
├── app.py                    # Streamlit web app
├── requirements.txt          # Dependencies
├── runtime.txt               # Python version
├── RandomForest_heaart.pkl   # Trained model
├── scaler.pkl                # Fitted StandardScaler
├── columns.pkl               # Column order
└── mappings.pkl              # Category-to-number mappings
```

---

## 🚀 Run Locally

```bash
# Clone the repo
git clone https://github.com/shubhdwi001/heart-disease-predictor.git
cd heart-disease-predictor

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## ⚠️ Disclaimer

This project is for **educational and demonstration purposes only**. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical decisions.

---

*Built by Sudhanshu Dwivedi as part of a machine learning learning journey.*
