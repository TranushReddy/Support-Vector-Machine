import os
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.svm import SVC, SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_squared_error,
    r2_score,
)
from datetime import datetime

# Page Config
st.set_page_config("END TO END SVM(BOTH REGRESSION AND CLASSIFICATION)", layout="wide")
st.title("END TO END SVM(BOTH REGRESSION AND CLASSIFICATION)")

# Sidebar
st.sidebar.header("Model Settings")
task = st.sidebar.selectbox("Task Type", ["Classification", "Regression"])
kernel = st.sidebar.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"])
C = st.sidebar.slider("C", 0.01, 10.0, 1.0)
gamma = st.sidebar.selectbox("Gamma", ["scale", "auto"])

# Folder setup
BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, "data/raw")
CLEAN = os.path.join(BASE, "data/cleaned")

os.makedirs(RAW, exist_ok=True)
os.makedirs(CLEAN, exist_ok=True)

# Step 1: Upload CSV
st.header("Step 1: Upload Dataset")
file = st.file_uploader("Upload ANY CSV file", type=["csv"])

df = None
if file:
    raw_path = os.path.join(RAW, file.name)
    with open(raw_path, "wb") as f:
        f.write(file.getbuffer())
    df = pd.read_csv(raw_path)
    st.success("Dataset uploaded successfully")
    st.dataframe(df.head())

# Step 2: EDA
if df is not None:
    st.header("Step 2: EDA")
    st.write("Shape:", df.shape)
    st.write("Missing values:", df.isnull().sum())

    num_cols = df.select_dtypes(include=[np.number])
    if not num_cols.empty:
        fig, ax = plt.subplots()
        sns.heatmap(num_cols.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

# Step 3: Data Cleaning
if df is not None:
    st.header("Step 3: Data Cleaning")

    df_clean = df.copy()

    for col in df_clean.columns:
        if df_clean[col].dtype == "object":
            df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
        else:
            df_clean[col].fillna(df_clean[col].mean(), inplace=True)

    st.session_state.df_clean = df_clean
    st.success("Missing values handled automatically")

# Step 4: Save Cleaned Data
if st.button("Save Cleaned Dataset"):
    if "df_clean" in st.session_state:
        name = f"cleaned_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        path = os.path.join(CLEAN, name)
        st.session_state.df_clean.to_csv(path, index=False)
        st.success(f"Saved at {path}")

# Step 5: Load Cleaned Dataset
st.header("Step 5: Load Cleaned Dataset")

files = os.listdir(CLEAN)
df_model = None
if files:
    selected = st.selectbox("Select Cleaned File", files)
    df_model = pd.read_csv(os.path.join(CLEAN, selected))
    st.dataframe(df_model.head())

# Step 6: Train SVM
if df_model is not None:
    st.header("Step 6: Train SVM Model")

    target = st.selectbox("Select Target Column", df_model.columns)

    X = df_model.drop(columns=[target])
    y = df_model[target]

    # Encode categorical features
    for col in X.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    # Encode target if classification
    if task == "Classification" and y.dtype == "object":
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Keep numeric only
    X = X.select_dtypes(include=[np.number])

    # Scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Train model
    if task == "Classification":
        model = SVC(kernel=kernel, C=C, gamma=gamma)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        st.success(f"Accuracy: {acc:.3f}")

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

    else:
        model = SVR(kernel=kernel, C=C, gamma=gamma)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.success(f"MSE: {mse:.3f}")
        st.success(f"R2 Score: {r2:.3f}")

        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        st.pyplot(fig)

