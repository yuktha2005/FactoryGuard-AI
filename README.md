# 🏭 FactoryGuard AI

### Predictive Maintenance System using Machine Learning

FactoryGuard AI is an **end-to-end Machine Learning–driven predictive maintenance system** designed to analyze industrial sensor data and predict potential equipment failures before they occur.

The goal of this project is to **improve factory reliability, reduce downtime, and enhance operational safety** by using machine learning models trained on real-world machine operating parameters.

This project demonstrates how **Machine Learning, Explainable AI (XAI), and API-based deployment** can be combined to build **scalable and trustworthy industrial AI solutions**.

---

# 📌 Project Overview

Modern industrial environments generate large volumes of sensor data from machines.
FactoryGuard AI uses this data to **predict failure risks early**, allowing preventive maintenance instead of reactive repairs.

The system processes machine parameters such as:

* Temperature
* Pressure
* Vibration
* Rotational Speed
* Load Conditions
* Operational Cycles

By analyzing these parameters, the model can **predict the probability of equipment failure** and identify **risk factors affecting machine performance**.

---

# 🛠 Tech Stack

## Backend

* Python
* Flask API
* Scikit-learn
* XGBoost
* Joblib

## Data Processing

* Pandas
* NumPy

## Explainable AI

* SHAP (SHapley Additive Explanations)

## Visualization

* Matplotlib
* Seaborn

## Frontend

* HTML
* CSS
* JavaScript

---

# 🚀 Getting Started

## Prerequisites

Make sure the following tools are installed:

* Python 3.8+
* pip (Python package manager)
* Git

You can check Python version:

```bash
python --version
```

---

# 📥 Installation

Clone the repository

```bash
git clone https://github.com/yuktha2005/FactoryGuard-AI.git
```

Navigate into the project directory

```bash
cd FactoryGuard-AI
```

Install required dependencies

```bash
pip install -r requirements.txt
```

---

# ⚙ Configuration

Ensure the following model files are present in the **models/** directory:

* `factoryguard_xgb.pkl` – trained XGBoost model
* `feature_columns.pkl` – list of model feature columns

These files are used by the Flask application to generate predictions.

---

# ▶ Running the Application

Start the Flask server:

```bash
python app.py
```

Once the server starts, open your browser and visit:

```
http://localhost:5000
```

You will see the **FactoryGuard AI prediction interface**, where you can input machine parameters and obtain failure predictions.

---

# ✨ Features

* Predict **equipment failure probability**
* Analyze **industrial sensor data**
* **Machine learning model integration**
* **Explainable AI insights using SHAP**
* **Interactive web interface**
* **Flask API for prediction requests**
* **Risk factor identification**

---

# 📂 Project Structure

```
FactoryGuard-AI
│
├── models/                     # Saved machine learning models
│
├── src/                        # Core ML scripts and modeling code
│
├── static/                     # Frontend static files
│   ├── style.css
│   ├── script.js
│   └── index.html
│
├── templates/                  # HTML templates for Flask
│
├── app.py                      # Flask application entry point
├── config.py                   # Configuration settings
│
├── requirements.txt            # Project dependencies
├── sample.json                 # Sample input data
│
├── Week4.ipynb                 # Model development notebook
├── week2_modeling.py           # Modeling workflow
├── week3_XAI_interpretability.py  # Explainable AI analysis
│
├── TODO.md                     # Development roadmap
└── .gitignore
```

---

# 🔮 Building Predictions

The prediction pipeline works as follows:

1. Sensor data is collected from the input form.
2. Data is preprocessed and aligned with **trained feature columns**.
3. The **XGBoost model** processes the data.
4. The system outputs:

   * Failure probability
   * Top contributing risk factors
5. SHAP values provide **model interpretability and transparency**.

---

# 🤖 Models & Algorithms Used

The project experiments with multiple machine learning algorithms:

* Random Forest Classifier
* XGBoost (Extreme Gradient Boosting)

### 🏆 Best Model

**XGBoost** was selected due to its:

* High predictive accuracy
* Robustness with structured sensor data
* Strong performance in industrial prediction tasks

---

# 🤝 Contributing

Contributions are welcome!

If you'd like to improve this project:

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Submit a pull request

Example:

```bash
git checkout -b feature-new-improvement
```

---

# 📜 License

This project is licensed under the **MIT License**.

---

# 🙏 Acknowledgment

I would like to sincerely thank **Infotact Solutions** for providing the opportunity to work on this project and gain hands-on experience in **Machine Learning, Explainable AI, and industrial predictive analytics**.

---

⭐ If you found this project useful, please consider **starring the repository**.
