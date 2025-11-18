# 🍽️ School Meal Analytics & Forecasting System (FCPS)

An AI-powered food service analytics platform designed to help Fairfax County Public Schools (FCPS) improve forecasting accuracy, reduce food waste, optimize production, and reduce operational costs using Machine Learning, LSTM/GRU deep learning models, XGBoost, and an interactive Streamlit dashboard.

---

## 🏷️ Badges

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10-blue" />
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C" />
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B" />
  <img src="https://img.shields.io/badge/XGBoost-Gradient%20Boosting-orange" />
  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-150458" />
  <img src="https://img.shields.io/badge/Numpy-Scientific%20Computing-013243" />
  <img src="https://img.shields.io/badge/BeautifulSoup-HTML%20Parsing-1B95E0" />
  <img src="https://img.shields.io/badge/Scikit--Learn-ML%20Models-F7931E" />
  <img src="https://img.shields.io/badge/GitHub-Version%20Control-181717" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
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

## 🔄 Dataset Workflow

Your pipeline transforms raw FCPS Production Records + POS data into a clean, structured forecasting dataset.

---

### **1. HTML → CSV Parser (`preprocess_html.py`)**

✔ Reads messy FCPS breakfast & lunch HTML files  
✔ Detects school blocks automatically  
✔ Extracts production, leftover, served, planned, discarded values  
✔ Cleans currencies, percentages, and item names  
✔ Standardizes headers  

**Outputs:**


