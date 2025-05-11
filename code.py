import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import joblib
import os

st.set_page_config(layout="wide")
st.title("🧠 ML Dashboard with Visualizations")

@st.cache_resource
def load_model():
    model_path = "model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

def get_predictions(data):
    if model is not None:
        try:
            return model.predict(data)
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            return None
    else:
        st.warning("⚠️ No model file found. Upload a model.pkl to enable predictions.")
        return None

uploaded_file = st.file_uploader("📤 Upload your CSV file", type=["csv"])

if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)
        st.success("✅ File loaded successfully!")
        st.write("### 🔍 Preview of Data", data.head())

        # Select ID column automatically or specify
        id_column = None
        for col in data.columns:
            if 'id' in col.lower():
                id_column = col
                break

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

        numeric_cols = data.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_cols = data.select_dtypes(include=["object", "category"]).columns.tolist()

        if numeric_cols:
            st.subheader("📈 Line Chart (first 100 rows)")
            st.line_chart(data[numeric_cols].head(100))

            st.subheader("📊 Correlation Heatmap")
            corr = data[numeric_cols].corr()
            fig, ax = plt.subplots()
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
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

            st.subheader("📍 Scatter Plot")
            if len(numeric_cols) >= 2:
                fig, ax = plt.subplots()
                sns.scatterplot(x=numeric_cols[0], y=numeric_cols[1], data=data, ax=ax)
                ax.set_title(f"Scatter Plot: {numeric_cols[0]} vs {numeric_cols[1]}")
                st.pyplot(fig)

            st.subheader("🧪 Bubble Chart")
            if len(numeric_cols) >= 3:
                fig, ax = plt.subplots()
                sns.scatterplot(x=numeric_cols[0], y=numeric_cols[1], size=numeric_cols[2],
                                data=data, ax=ax, legend=False, sizes=(20, 200))
                ax.set_title(f"Bubble Chart: {numeric_cols[0]} vs {numeric_cols[1]} sized by {numeric_cols[2]}")
                st.pyplot(fig)

            st.subheader("🕸 Spider/Radar Chart")
            if len(numeric_cols) >= 3:
                from math import pi
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
            st.write("### 📋 Model Predictions")
            st.write(predictions)

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
else:
    st.info("📎 Please upload a CSV file to begin.")