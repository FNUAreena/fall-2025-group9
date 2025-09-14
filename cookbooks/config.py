# cookbooks/config.py
import os
BASE = os.environ.get("FCPS_BASE", "/content/drive/MyDrive/FCPS")

# ===== Inputs from Google Drive =====
LUNCH_HTML_DIR     = f"{BASE}/May 2025 Lunch production records/May 2025 Lunch production records"
BREAKFAST_HTML_DIR = f"{BASE}/May 2025 Breakfast production records/May 2025 Breakfast production records"
POS_PDF_DIR        = f"{BASE}/Item Sales Reports - Mar May 2025/Item Sales Reports - Mar May 2025"

# ===== Outputs back to Google Drive =====
PREPROC_LUNCH_DIR     = f"{BASE}/preprocess/html-processing/preprocessed-data/Lunch production"
PREPROC_BREAKFAST_DIR = f"{BASE}/preprocess/html-processing/preprocessed-data/Breakfast production"
POS_OUT_DIR           = f"{BASE}/preprocess/pos-processing/preprocessed-data"

for p in [PREPROC_LUNCH_DIR, PREPROC_BREAKFAST_DIR, POS_OUT_DIR]:
    os.makedirs(p, exist_ok=True)
