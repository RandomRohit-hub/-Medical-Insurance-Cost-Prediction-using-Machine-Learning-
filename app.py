import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import os

st.set_page_config(page_title="Insurance Cost Predictor", layout="centered")
st.title("🏥 Insurance Cost Predictor 💰")

# Automatically use the CSV file located in the same directory (used in Jupyter Notebook)
csv_file_path = "insurance.csv"

if os.path.exists(csv_file_path):
    df = pd.read_csv(csv_file_path)
    st.sidebar.success(f"📄 Using dataset: {csv_file_path}")

    st.subheader("🔍 Data Preview")
    st.dataframe(df.head())

    st.subheader("📊 Dataset Info")
    st.write(df.describe())

    st.subheader("🚨 Missing Values")
    st.write(df.isnull().sum())

    st.subheader("📈 Visualizations")
    col = st.selectbox("Select column to visualize", df.columns)
    if df[col].dtype in ["int64", "float64"]:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        st.pyplot(fig)
    else:
        fig, ax = plt.subplots()
        sns.countplot(x=col, data=df, ax=ax)
        st.pyplot(fig)

    # Data Preprocessing
    df = pd.get_dummies(df, drop_first=True)
    X = df.drop('charges', axis=1)
    y = df['charges']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    st.subheader("📉 Model Evaluation")
    st.write("R2 Score:", metrics.r2_score(y_test, y_pred))
    st.write("MAE:", metrics.mean_absolute_error(y_test, y_pred))

    st.subheader("🧮 Make a Prediction")
    with st.form("predict_form"):
        input_data = {}
        for i, column in enumerate(X.columns):
            widget_key = f"input_{i}_{column}"
            if column == "sex_male":
                input_data[column] = st.selectbox("Sex 🧑‍⚕️", ["female", "male"], key=widget_key) == "male"
            elif column == "smoker_yes":
                input_data[column] = st.selectbox("Smoker 🚬", ["no", "yes"], key=widget_key) == "yes"
            elif column.startswith("region_"):
                continue  # Skip dummy region fields
            else:
                input_data[column] = st.number_input(column, step=1.0, key=widget_key)

        # Add region dropdown separately
        regions = ['northeast', 'northwest', 'southeast', 'southwest']
        selected_region = st.selectbox("Region 🌎", regions)
        for region in regions[1:]:  # drop_first=True skips the first region (e.g. northeast)
            col_name = f"region_{region}"
            input_data[col_name] = int(region == selected_region)

        submitted = st.form_submit_button("Predict 💡")

        if submitted:
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)[0]
            st.success(f"🎯 Predicted Insurance Cost: ${prediction:.2f}")

else:
    st.error("❌ CSV file 'insurance.csv' not found in the current directory. Please ensure it is placed alongside this script.")