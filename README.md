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

Your pipeline transforms raw FCPS Production Records + POS data → **clean, structured forecasting dataset**.

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
cd src/univariate
python main.py
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
cd src/multivariate
python main.py
```
Uses features:
served_total
planned_total
discarded_total
left_over_total
And predicts:
production_cost_total
Models saved to:

```
multivariate/results/
multivariate/plots/
```

**4. Model Comparison**

```
cd src/univariate
python comparing_model.py
```
This evaluates:
Linear Regression
XGBoost
Feed-Forward Neural Network
LSTM
GRU
Outputs saved into:

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

