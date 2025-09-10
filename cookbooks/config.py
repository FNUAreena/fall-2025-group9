# cookbooks/config.py
import os
BASE = os.environ.get("FCPS_BASE", "/content/drive/MyDrive/Capstone Project/Data")

# INPUTS (EDIT folder names if different)
LUNCH_HTML_DIR     = f"{BASE}/May 2025 Lunch production records/May 2025 Lunch production records"
BREAKFAST_HTML_DIR = f"{BASE}/May 2025 Breakfast production records/May 2025 Breakfast production records"
POS_PDF_DIR        = f"{BASE}/Item Sales Reports - Mar May 2025/Item Sales Reports - Mar May 2025"
MENUS_DIR          = f"{BASE}/Menus"

# OUTPUTS (written back to Drive)
PREPROC_LUNCH_DIR     = f"{BASE}/preprocess/html-processing/preprocessed-data/Lunch production"
PREPROC_BREAKFAST_DIR = f"{BASE}/preprocess/html-processing/preprocessed-data/Breakfast production"
POS_OUT_DIR           = f"{BASE}/preprocess/pos-processing/preprocessed-data"
MERGE_OUT_DIR         = f"{BASE}/preprocess/merge/preprocessed-data"
MODELS_OUT_DIR        = f"{BASE}/models/outputs"
DASHBOARD_DATA_DIR    = f"{BASE}/dashboard-data"

for p in [PREPROC_LUNCH_DIR, PREPROC_BREAKFAST_DIR, POS_OUT_DIR, MERGE_OUT_DIR, MODELS_OUT_DIR, DASHBOARD_DATA_DIR]:
    os.makedirs(p, exist_ok=True)
