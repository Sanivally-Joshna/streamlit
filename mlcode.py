import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import joblib
import os
from math import pi
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(layout="wide")
st.title("Machine Learning Dashboard with Enhanced Visuals")

with st.sidebar:
    st.title("Project Information")
    st.markdown("""
    **Goal**: Predict high potential customers using ML  
    **Model**: Trained with customer data  
    **Input**: Demographic and financial features  
    **Output**: Prediction (0 = Low Potential, 1 = High Potential)
    """)

@st.cache_resource
def load_model():
    model_path = "model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

uploaded_model = st.file_uploader("Upload your trained model.pkl", type=["pkl"])

@st.cache_resource
def load_model_from_upload(uploaded_file):
    if uploaded_file is not None:
        return joblib.load(uploaded_file)
    return None

model = load_model_from_upload(uploaded_model) or load_model()

expected_features = ['CustomerID', 'Gender', 'EmploymentStatus', 'CreditScore',
                     'AnnualIncome', 'LoanAmount', 'ExistingLoans', 'Purpose',
                     'LoanHistory', 'Savings', 'HighPotentialCustomer']

@st.cache_data
def preprocess_data(df):
    df = df.copy()
    for col in df.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df

def get_predictions(data):
    if model is not None:
        try:
            input_data = data[expected_features].copy()
            input_data = preprocess_data(input_data)
            return model.predict(input_data), model.predict_proba(input_data)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return None, None
    else:
        st.warning("No model loaded.")
        return None, None

uploaded_file = st.file_uploader("Upload your CSV data file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("Data loaded successfully!")
        st.write("### Data Preview", df.head())

        st.subheader("Dataset Summary")
        st.markdown(f"- Rows: {df.shape[0]}  \n- Columns: {df.shape[1]}")
        st.markdown(f"- Missing Values: {df.isnull().sum().sum()}")

        id_column = next((col for col in df.columns if 'id' in col.lower()), None)
        if id_column:
            st.subheader("Search by ID")
            search_id = st.text_input(f"Enter {id_column} value")
            if search_id:
                match = df[df[id_column].astype(str) == search_id]
                if not match.empty:
                    st.write("### Details:", match)
                else:
                    st.warning("ID not found.")

        num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        st.subheader("Descriptive Statistics")
        st.dataframe(df.describe(include='all').T.style.background_gradient(cmap='twilight'))

        if len(num_cols) >= 2:
            st.subheader("Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(df[num_cols].corr(), annot=True, cmap='mako', fmt=".2f", linewidths=0.5, ax=ax)
            st.pyplot(fig)

        st.subheader("Histograms")
        for col in num_cols:
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, color='#4B0082', ax=ax)  # Dark Purple color
            ax.set_title(f"Distribution of {col}", fontsize=14)
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")
            ax.set_facecolor('#f0f0f0')
            st.pyplot(fig)

        st.subheader("Categorical Columns - Top Categories")
        for col in cat_cols:
            fig, ax = plt.subplots()
            top = df[col].value_counts().nlargest(5)
            sns.barplot(x=top.values, y=top.index, ax=ax, palette="dark")
            ax.set_title(f"Most Frequent Categories in {col}", fontsize=14)
            ax.set_xlabel("Count")
            ax.set_ylabel(col)
            st.pyplot(fig)

        if len(num_cols) >= 2:
            st.subheader("Scatter Plot")
            fig, ax = plt.subplots()
            sns.scatterplot(x=num_cols[0], y=num_cols[1], hue=df[cat_cols[0]] if cat_cols else None,
                            palette='dark', data=df, ax=ax)
            ax.set_title(f"Scatter: {num_cols[0]} vs {num_cols[1]}")
            st.pyplot(fig)

            st.subheader("Area Chart")
            st.area_chart(df[num_cols].head(100))

        if len(num_cols) >= 5:
            st.subheader("Radar Chart of Mean Features")
            categories = num_cols[:5]
            values = df[categories].mean().tolist()
            values += values[:1]
            angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
            angles += angles[:1]
            fig, ax = plt.subplots(subplot_kw={'polar': True})
            ax.plot(angles, values, linewidth=2, color='#8A2BE2')  # Violet color
            ax.fill(angles, values, color='#8A2BE2', alpha=0.3)
            plt.xticks(angles[:-1], categories)
            st.pyplot(fig)

        st.markdown("""<h2 style='color:#8A2BE2; font-size:28px;'>Predictions</h2>""", unsafe_allow_html=True)
        predictions, proba = get_predictions(df)
        if predictions is not None:
            df["Prediction"] = predictions
            st.write("### Predictions")
            st.dataframe(df)

            st.write("### Prediction Class Distribution")
            fig, ax = plt.subplots()
            sns.countplot(x="Prediction", data=df, palette="mako", ax=ax)
            ax.set_title("Predicted Class Distribution", fontsize=14)
            st.pyplot(fig)

            high_risk = df[df["Prediction"] == 1]
            st.subheader("High Potential Customers")
            st.dataframe(high_risk.head())

            st.subheader("Classification Report")
            true_labels = df["HighPotentialCustomer"] if "HighPotentialCustomer" in df.columns else None
            if true_labels is not None:
                report = classification_report(true_labels, predictions, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose())

                st.subheader("Confusion Matrix")
                cm = confusion_matrix(true_labels, predictions)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Low", "High"], yticklabels=["Low", "High"], ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)

            @st.cache_data
            def convert_df(df):
                return df.to_csv(index=False).encode('utf-8')

            st.download_button("Download Predictions", convert_df(df), "predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"Error loading data: {e}")
else:
    st.info("Upload a CSV file to explore your data and get predictions.")
