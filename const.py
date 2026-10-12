import os
from os.path import expanduser


ROOT = expanduser("~")
DATA_FOLDER = "Data"
REPO_PATH = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(ROOT, DATA_FOLDER, "sent_risk_pred")

SAFERCAR_DATA_PATH = os.path.join(DATA_PATH, "Safercar_data.csv")
FLAT_CMPL_PATH = os.path.join(DATA_PATH, "FLAT_CMPL.txt")
RCL_FROM_2010_2014_PATH = os.path.join(DATA_PATH, "RCL_FROM_2010_2014.csv")
RCL_FROM_2015_2019_PATH = os.path.join(DATA_PATH, "RCL_FROM_2015_2019.csv")
RCL_FROM_2020_2024_PATH = os.path.join(DATA_PATH, "RCL_FROM_2020_2024.csv")
RCL_FROM_2025_2025_PATH = os.path.join(DATA_PATH, "RCL_FROM_2025_2025.csv")
RCL_FROM_2025_2026_PATH = os.path.join(DATA_PATH, "RCL_FROM_2025_2026.csv")
RCL_DATA_PATTERN = os.path.join(DATA_PATH, "RCL_FROM_*.csv")

INTERM_RES = os.path.join(DATA_PATH, "interm")
CLEANED_COMPLAINTS_PATH = os.path.join(INTERM_RES, "cleaned_complaints.parquet")
CLEANED_RECALLS_PATH = os.path.join(INTERM_RES, "cleaned_recalls.parquet")
MATCHED_COMPLAINTS_PATH = os.path.join(INTERM_RES, "matched_complaints.parquet")
DATA_REPORT_PATH = os.path.join(INTERM_RES, "data_report.json")
COMPLAINT_FEATURES_PATH = os.path.join(INTERM_RES, "complaint_features.parquet")
VEHICLE_RISK_SUMMARY_PATH = os.path.join(
    INTERM_RES,
    "vehicle_risk_summary.parquet",
)
RECALL_RISK_SUMMARY_PATH = os.path.join(
    INTERM_RES,
    "recall_risk_summary.parquet",
)

FIGS_PATH = os.path.join(REPO_PATH, "figs")

SENTIMENT_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
SENTIMENT_BATCH_SIZE = 32
SENTIMENT_MAX_LENGTH = 512
FEATURE_INFERENCE_CHUNK_SIZE = 4096
FEATURE_SAMPLE_RANDOM_STATE = 42
