# Music Emotion Detection - Streamlit App

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier

import warnings
warnings.filterwarnings('ignore')

st.title("🎵 Music Emotion Detection")
st.write("Predicting mood using Machine Learning")

# Load data
data = pd.read_csv('data/data.csv')
data.columns = data.columns.str.strip()

st.subheader("Dataset Preview")
st.dataframe(data.head())

# Features
emotion_features = [
    'amazement', 'solemnity', 'tenderness', 'nostalgia', 'calmness',
    'power', 'joyful_activation', 'tension', 'sadness'
]

X = data[emotion_features]
y = data['mood']
st.write(y.value_counts())
# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Visualization
st.subheader("Average Emotion Scores")
plt.figure()
X.mean().plot(kind='bar')
plt.xticks(rotation=45)
st.pyplot(plt)

# Models
pipelines = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=500))
    ]),
    'PCA + Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=5)),
        ('clf', LogisticRegression(max_iter=500))
    ]),
    'Neural Network': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', MLPClassifier(max_iter=500))
    ]),
    'Random Forest': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=100))
    ])
}

results = {}
predictions = {}

st.subheader("Model Performance")

for name, model in pipelines.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[name] = y_pred

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

    results[name] = {
        'Accuracy': acc,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }

results_df = pd.DataFrame(results).T
st.dataframe(results_df)

# Confusion Matrix
st.subheader("Confusion Matrices")

for name, y_pred in predictions.items():
    st.write(f"### {name}")
    plt.figure()
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    st.pyplot(plt)

# Best model
best_model = results_df['Accuracy'].idxmax()

st.subheader("Best Model")
st.success(best_model)

st.subheader("Classification Report")
st.text(classification_report(y_test, predictions[best_model]))
