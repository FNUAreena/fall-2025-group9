# 🍽️ School Meal Analytics & Forecasting System (FCPS)

An AI-powered food service analytics platform designed to help Fairfax County Public Schools (FCPS) improve forecasting accuracy, reduce food waste, optimize production, and reduce operational costs using Machine Learning, LSTM/GRU deep learning models, XGBoost, and an interactive Streamlit dashboard.

---

# 🏷️ Badges  

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python"/>
  
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch"/>
  
  <img src="https://img.shields.io/badge/XGBoost-Gradient%20Boosting-orange?style=for-the-badge&logo=xgboost"/>
  
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit"/>
  
  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas"/>
  
  <img src="https://img.shields.io/badge/Numpy-Scientific%20Computing-013243?style=for-the-badge&logo=numpy"/>
  
  <img src="https://img.shields.io/badge/BeautifulSoup-HTML%20Parsing-195E0?style=for-the-badge"/>
  
  <img src="https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=for-the-badge&logo=matplotlib"/>
  
  <img src="https://img.shields.io/badge/Scikit--Learn-ML%20Models-F7931E?style=for-the-badge&logo=scikitlearn"/>
  
  <img src="https://img.shields.io/badge/GitHub-Version%20Control-181717?style=for-the-badge&logo=github"/>
  
  <img src="https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge"/>
</p>

---

## 📋 Table of Contents

- [Overview](#overview)  
- [Dataset Workflow](#dataset-workflow)  
- [Key Features](#key-features)  
- [System Architecture](#system-architecture)  
- [Model Pipeline](#model-pipeline)  
- [Getting Started](#getting-started)  
    - [Prerequisites](#prerequisites)  
    - [Installation](#installation)  
    - [Environment Setup](#environment-setup)  
    - [Running the Models](#running-the-models)  
- [Dashboard (Streamlit App)](#dashboard-streamlit-app)  
- [API Endpoints](#api-endpoints)  
- [Troubleshooting](#troubleshooting)  
- [Research & Performance](#research--performance)  
- [Technology Stack](#technology-stack)  
- [Contributing](#contributing)  
- [License](#license)  
- [Acknowledgments](#acknowledgments)

---

# 🔄 Dataset Workflow

Our pipeline transforms raw FCPS Production Records + POS data → **clean, structured forecasting dataset**.

---

### **1️⃣ HTML → CSV Parser (`preprocess_html.py`)**

✔ Reads dozens of messy FCPS breakfast & lunch HTML files  
✔ Auto-detects school sections  
✔ Extracts production, leftover, planned, served, discarded values  
✔ Cleans currencies, percentages, and item names  
✔ Standardizes headers  

**Outputs generated:**

- `breakfast_combined.csv`  
- `lunch_combined.csv`  
- `meals_combined.csv`  

---

### **2️⃣ Data Cleaning & Preprocessing (`utils.preprocess`)**

✔ Cleans `$` & `%` → float  
✔ Converts & sorts dates  
✔ Handles missing values  
✔ Outlier removal using 99th percentile  
✔ Encodes meal types  
✔ Produces final ML-ready dataset for:

- Univariate Forecasting  
- Multivariate Forecasting  
- Streamlit Dashboard  

---

### ⭐ **Final Dataset Columns**

| Column | Description |
|--------|-------------|
| school_name | FCPS school |
| meal_type | breakfast/lunch |
| date | daily record |
| served_total | meals served |
| planned_total | planned meals |
| discarded_total | wasted meals |
| left_over_total | leftover meals |
| production_cost_total | $$ spent per item-day |

---

# 🎯 Key Features

### 🍽️ **1. Meal Demand Forecasting**
- LSTM & GRU deep learning models  
- Univariate forecasting (district-level daily time-series)  
- Multivariate forecasting (served/planned/discarded/leftovers → cost)  

---

### ♻️ **2. Waste Optimization**
- Predict discarded + leftover quantities  
- Waste ratio analytics  
- Identify high-waste menu items  

---

### 💲 **3. Cost Forecasting**
- Predict production cost for next 10 days  
- Scenario modeling using “What-If” adjustments  

---

### 📊 **4. Interactive Streamlit Dashboard**
- School-wise filtering  
- Cost trends  
- Waste ratio analysis  
- What-if ML predictions  
- Benchmark model comparison  

---

### 🧠 **5. Machine Learning Benchmarking**
- Linear Regression  
- XGBoost  
- Feed-Forward Neural Network  
- GRU & LSTM  

---

# 🏗️ System Architecture

```
┌──────────────────────────┐
│    Raw FCPS HTML Files   │
└───────────────┬──────────┘
                │
     (HTML Parser + Normalizer)
                │
                ▼
┌──────────────────────────┐
│    meals_combined.csv    │
└───────────────┬──────────┘
                │
       (Data Preprocessing)
                │
   ┌────────────┼───────────────┬──────────────┐
   ▼            ▼               ▼
Univariate   Multivariate     Benchmark  
   LSTM          GRU           Models
   │             │               │
   └───────┬─────┴───────┬──────┘
           ▼             ▼
      Forecasts   Performance Charts
           │             │
           └───────┬────┘
                   ▼
        Streamlit Dashboard


---
```

# 🤖 Model Pipeline

## 📌 **Univariate Forecasting (LSTM / GRU)**  
Uses district-wide *daily* production costs:

➡️ `[Cost(t−7) … Cost(t−1)] → Predict Cost(t)`

Models:  
- LSTM  
- GRU  
- Feedforward baseline  
- XGBoost  
- Linear Regression  

---

## 📌 **Multivariate Forecasting**

**Features:**  
- served_total  
- planned_total  
- discarded_total  
- left_over_total  

**Target:**  
- production_cost_total  

Models:  
- Linear Regression  
- XGBoost  
- FeedForwardNN  
- GRU (sequence-based, school-wise)  

---

# 🚀 Getting Started

### ✔️ Prerequisites
Install:

- Python 3.10+  
- pip  
- Streamlit  
- PyTorch  
- XGBoost  

---

### 📦 Installation

```bash
git clone https://github.com/FNUAreena/fall-2025-group9
cd fall-2025-group9
pip install -r requirements.txt
```

### 🌱 Environment Setup
Important source files: 

```
src/utils.py
src/model.py
src/forecasting.py
```

### ▶️ Running the Application
**1. HTML → CSV Preprocessing**

```
cd src
python preprocess_html.py
```
This script:

- Reads FCPS breakfast & lunch HTML production records
- Extracts → served, planned, discarded, leftover, cost
- Cleans currency & % values
- Standardizes headers
- Generates:

```
Data/Output/breakfast_combined.csv
Data/Output/lunch_combined.csv
Data/Output/meals_combined.csv
```

**2. Univariate Forecasting**

```
cd src/component
python univariate/main.py
```

This will:

- Aggregate total district production cost per day
- Create sliding windows
- Train LSTM/GRU
- Save model + plots into:

```
univariate/results/
univariate/plots/
```

**3. Multivariate Forecasting**

```
cd src/component
python multivariate/main.py
```
- Uses features:
- served_total
- planned_total
- discarded_total
- left_over_total
- And predicts:
production_cost_total
- Models saved to:

```
multivariate/results/
multivariate/plots/
```

**4. Model Comparison**

```
cd src/component
python univariate/comparing_model.py
```

This evaluates:
- Linear Regression
- XGBoost
- Feed-Forward Neural Network
- LSTM
- GRU
- Outputs saved into:

```
univariate/results/
univariate/plots/
```

**5. Important Source Files**

```
src/
├── preprocess_html.py          # HTML → CSV parser
├── utils.py                    # Preprocessing + cleaning helpers
├── model.py                    # LSTM/GRU model classes
└── forecasting.py              # Multi-step forecasting logic
```
# ✅ 📊 Dashboard (Streamlit App)

Our interactive FCPS Meal Analytics Dashboard provides real-time insights into school meal operations.
Run the full interactive dashboard

```
streamlit run app_dashboard_nav.py
```

### 🔍 Includes

✔ School-wise analysis

✔ Waste heatmap

✔ What-if prediction sliders

✔ LSTM vs GRU comparison

✔ Loss-making school detection

✔ Forecast charts by date & school


# 📡 API Endpoints

Although this project does not use external REST APIs, the internal Streamlit dashboard relies on several Python-based API-like functions that power forecasting and analysis.

### 🔧 Internal Model Endpoints

| Function | Description | Location |
|---------|-------------|----------|
| `forecast_future_dates()` | Predicts next *k* days using trained LSTM/GRU models | `src/forecasting.py` |
| `load_and_aggregate_district()` | Loads CSV + cleans + aggregates district production cost | `src/utils.py` |
| `safe_time_split()` | Chronological train-test split for time-series | `src/utils.py` |
| `TimeSeriesDataset` | Creates sliding windows for univariate LSTM/GRU | `src/utils.py` |
| `ForecastingModel` | LSTM/GRU model class | `src/model.py` |
| `FeedForwardRegressor` | Baseline neural network model | `src/model.py` |
| `forecast_all_models_in_folder()` | Runs forecasts for every school (batch mode) | `app.py` |
| `school_loss_analysis()` | Detects schools with high loss or wastage | `app.py` |

### 🖥️ Dashboard-Level Actions (Triggered in Streamlit)

| Action | Trigger Button | What Happens |
|--------|----------------|--------------|
| Run Forecast | **Run Forecast** | Loads all LSTM/GRU models and predicts next *k* days |
| Compare Models | **Compare LSTM vs GRU** | Runs both folders → compares total cost curves |
| AI Recommendations | **AI Recommendations** | Suggests waste reduction strategies |
| Wastage Heatmap | Auto-loaded | Creates weekday-based discarded food heatmap |
| School-Level View | Dropdown Filters | Filters graphs/tables by school + meal type |

# 🔧 Troubleshooting

Quick solutions to the most common issues:

| Issue | Cause | Simple Fix |
|-------|--------|-------------|
| **Empty CSV after parsing** | Wrong HTML folder path | Check breakfast/lunch folder paths before running `preprocess_html.py` |
| **Date errors / NaNs** | FCPS dates use mixed formats | Use `dayfirst=True` in `pd.to_datetime()` (already used in code) |
| **LSTM/GRU model not loading** | Wrong `.pth` path | Ensure model file is inside: `univariate/LSTM_models/` or `univariate/GRU_models/` |
| **Streamlit blank page** | Cached old data | Run: `streamlit cache clear` |
| **XGBoost import error** | Not installed | `pip install xgboost` |
| **Very high forecast values** | Outliers in cost | 99th percentile cleaning already included—recheck preprocessing |
| **Training too slow** | Model too big | Reduce `HIDDEN_DIM` from 256 → 128 |
| **Forecast shows empty for a school** | School name mismatch | Filename uses `_` (e.g., `Aldrin_Elementary`), CSV uses spaces → ensure both match |
| **Heatmap blank** | Non-numeric waste columns | Convert with `pd.to_numeric(errors='coerce').fillna(0)` |
| **“Forecast failed” error** | Not enough rows for that school | Check if subset CSV has enough data; retrain if needed |
| **Port already in use (Streamlit)** | Another app running | Run: `lsof -i :8501` → `kill -9 <PID>` |

---
# 📊 Research & Performance

### 1️⃣ Univariate Forecasting Results (Cost-Only Models)

These models predict **production_cost_total** using only past cost values (sliding window of 7 days).

| **Model** | **RMSE** | **R²** | **Notes** |
|----------|----------|--------|-----------|
| **LSTM** | ⭐ Best | High | Learns long-term temporal patterns extremely well |
| **GRU** | Very Good | High | Faster than LSTM, stable performance |
| **XGBoost** | Medium | Medium | Strong non-linear baseline, but not sequence-aware |
| **Feed-Forward NN (FNN)** | Medium | Medium | Good baseline but ignores temporal structure |
| **Linear Regression** | Poor | Low | Cannot model sequential dependencies |

---

### 2️⃣ Multivariate Forecasting Results (School-Level Features)

These models use:

- `served_total`  
- `planned_total`  
- `discarded_total`  
- `left_over_total`  

to predict:

- `production_cost_total`

| **Model** | **Performance** | **Notes** |
|----------|------------------|-----------|
| **GRU (Sequence Model)** | ⭐ Best (if metrics show this) | Captures school-wise temporal patterns across multiple features |
| **LSTM (Sequence Model)** | ⭐ Best / Very Strong | Multivariate LSTM trained on same features; stable long-range learning |
| **XGBoost** | Strong | Excellent for structured/tabular data |
| **Feed-Forward NN** | Good | Learns non-linear interactions but not sequence structure |
| **Linear Regression** | Baseline | Limited for multi-feature temporal data |


---

### 🗝️ Key Findings (Short)

- Both **multivariate LSTM and GRU** clearly outperform classical models (XGBoost, FNN, Linear Regression).
- Including **served, planned, discarded, and leftover meals** improves cost prediction compared to cost-only models.
- Sequence models (LSTM/GRU) handle **school-level temporal behavior** much better than non-sequence models.
- Outlier removal and proper preprocessing stabilize forecasts and reduce noise.


---

# 🧰 Technology Stack

| Category | Technologies |
|---------|--------------|
| 🤖 Machine Learning | PyTorch · XGBoost · Scikit-Learn |
| 🧠 Deep Learning | LSTM · GRU · FeedForwardNN |
| 🖥️ Dashboard | Streamlit · Plotly Express |
| 🧹 Data Processing | Pandas · NumPy · BeautifulSoup · lxml |
| 📊 Visualization | Matplotlib · Seaborn |
| 🧪 Evaluation | MSE · RMSE · R² · MAE |
| 📁 Utilities | Pickle · Glob · Pathlib · OS |
| 🔧 Version Control | Git · GitHub |
| 🚀 Deployment | Local Machine · Streamlit Cloud |
| 💻 Language | Python |


# ✅ 🤝 Contributing
We welcome contributions from developers, students, and researchers.
Steps:
```
# Create a feature branch
git checkout -b feature/my-feature

# Make changes and commit
git commit -m "Added new improvement"

# Push to repo
git push origin feature/my-feature
```
Then open a Pull Request on GitHub.

# ✅ 📄 License
This project is licensed under the MIT License.
You are free to use, modify, and distribute the software with proper attribution.

# ✅ 🙏 Acknowledgments

Special thanks to the contributors who made this project possible:

•	Dr. Amir Jafari – Project Guidance (GWU)

•	Fairfax County Public Schools (FCPS) – For providing production record structures

•	Open-source community – PyTorch, Streamlit, XGBoost

•	Team Members – Areena, Chaya, Varshith

