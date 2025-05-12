import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import joblib
import os
from math import pi
from sklearn.preprocessing import LabelEncoder

st.set_page_config(layout="wide")
st.title("🧠 ML Dashboard with Visualizations")

# Load model from local fallback
@st.cache_resource
def load_model():
    model_path = "model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

# Load uploaded model
uploaded_model = st.file_uploader("📦 Upload your trained model.pkl", type=["pkl"])

@st.cache_resource
def load_model_from_upload(uploaded_file):
    if uploaded_file is not None:
        return joblib.load(uploaded_file)
    return None

# Use uploaded model if available, fallback to local
model = load_model_from_upload(uploaded_model) or load_model()

# Function to preprocess categorical columns
def preprocess_data(df):
    df = df.copy()
    for col in df.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df

# Prediction function
# Replace this with the actual features used when training the model
expected_features = [ 'CustomerID',  'Gender',   'EmploymentStatus',
  'CreditScore', 'AnnualIncome', 'LoanAmount',  'ExistingLoans',
  'Purpose', 'LoanHistory', 'Savings', 'HighPotentialCustomer']

def get_predictions(original_data):
    if model is not None:
        try:
            input_data = original_data[expected_features].copy()
            input_data = preprocess_data(input_data)
            predictions = model.predict(input_data)
            return predictions
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            return None
    else:
        st.warning("⚠️ No model loaded.")
        return None

# Upload CSV file
uploaded_file = st.file_uploader("📤 Upload your CSV file", type=["csv"])

if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)
        st.success("✅ File loaded successfully!")
        st.write("### 🔍 Preview of Data", data.head())

        st.write("Uploaded data shape:", data.shape)
        st.write("Columns:", data.columns.tolist())

        # Auto-detect ID column
        id_column = next((col for col in data.columns if 'id' in col.lower()), None)

        if id_column:
            st.subheader("🔎 Search by ID")
            search_id = st.text_input(f"Enter a value from `{id_column}` to get full details")

            if search_id:
                matched_rows = data[data[id_column].astype(str) == str(search_id)]
                if not matched_rows.empty:
                    st.write("### 🔎 Details for ID:", search_id)
                    st.dataframe(matched_rows)
                else:
                    st.warning("⚠️ No matching ID found.")

        # Separate columns
        numeric_cols = data.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_cols = data.select_dtypes(include=["object", "category"]).columns.tolist()

        if numeric_cols:
            st.subheader("📈 Line Chart (first 100 rows)")
            st.line_chart(data[numeric_cols].head(100))

            st.subheader("📊 Correlation Heatmap")
            fig, ax = plt.subplots()
            sns.heatmap(data[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

            st.subheader("📌 Histograms")
            for col in numeric_cols:
                fig, ax = plt.subplots()
                sns.histplot(data[col], kde=True, ax=ax)
                ax.set_title(f"Histogram of {col}")
                st.pyplot(fig)

            st.subheader("📊 Bar Plots (Top 5 Frequent Categories)")
            for col in categorical_cols:
                fig, ax = plt.subplots()
                top_categories = data[col].value_counts().nlargest(5)
                sns.barplot(x=top_categories.values, y=top_categories.index, ax=ax)
                ax.set_title(f"Top 5 Categories in '{col}'")
                ax.set_xlabel("Count")
                st.pyplot(fig)

            st.subheader("📉 Area Chart (first 100 rows)")
            st.area_chart(data[numeric_cols].head(100))

            if len(numeric_cols) >= 2:
                st.subheader("📍 Scatter Plot")
                fig, ax = plt.subplots()
                sns.scatterplot(x=numeric_cols[0], y=numeric_cols[1], data=data, ax=ax)
                ax.set_title(f"Scatter Plot: {numeric_cols[0]} vs {numeric_cols[1]}")
                st.pyplot(fig)

            if len(numeric_cols) >= 3:
                st.subheader("🧪 Bubble Chart")
                fig, ax = plt.subplots()
                sns.scatterplot(x=numeric_cols[0], y=numeric_cols[1], size=numeric_cols[2],
                                data=data, ax=ax, legend=False, sizes=(20, 200))
                ax.set_title(f"Bubble Chart: {numeric_cols[0]} vs {numeric_cols[1]} sized by {numeric_cols[2]}")
                st.pyplot(fig)

                st.subheader("🕸 Spider/Radar Chart")
                categories = numeric_cols[:5]
                radar_data = data[categories].mean()
                N = len(categories)
                angles = [n / float(N) * 2 * pi for n in range(N)]
                angles += angles[:1]
                radar_values = radar_data.tolist()
                radar_values += radar_values[:1]

                fig, ax = plt.subplots(subplot_kw={'polar': True})
                plt.xticks(angles[:-1], categories)
                ax.plot(angles, radar_values)
                ax.fill(angles, radar_values, alpha=0.25)
                st.pyplot(fig)
        else:
            st.info("No numeric columns found for visualization.")

        st.subheader("🤖 Predictions")
        predictions = get_predictions(data)
        if predictions is not None:
            data["Prediction"] = predictions
            st.write("### 📋 Model Predictions with Data")
            st.dataframe(data)

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
else:
    st.info("📎 Please upload a CSV file to begin.")