
---

# 🏥 Insurance Cost Predictor 💰


https://8q2b8sdh7jkubabjusytaj.streamlit.app/




This Streamlit web application predicts insurance costs based on user inputs using a linear regression model trained on real-world insurance data

## 🚀 Features

- 📥 Automatically loads `insurance.csv` from the local directory
- 🔍 View data preview and statistics
- 📊 Visualize distributions and category counts
- 🧠 Train a linear regression model
- 🧮 Predict insurance costs with an interactive form
- 📈 View model performance metrics

## 📂 Dataset

The app expects a CSV file named `insurance.csv` in the same directory. The dataset should include features such as:

- `age`, `sex`, `bmi`, `children`, `smoker`, `region`, and `charges`

👉 **You can also change the CSV file used in the code** (`csv_file_path = "insurance.csv"`) to load a different dataset and see predictions based on your own data.

## 🛠 Installation

1. Clone this repository or copy the script.
2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Make sure your dataset CSV file is in the same folder.

## ▶️ Running the App

```bash
streamlit run app.py
```

## 📸 Screenshots

<img src="https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png" alt="Streamlit" width="150">

## 📦 Requirements

See `requirements.txt`:

```
streamlit
pandas
seaborn
matplotlib
scikit-learn
```

## 💡 Example Inputs

- Age: 29  
- Sex: Male  
- BMI: 27.9  
- Children: 0  
- Smoker: Yes  
- Region: Southwest

