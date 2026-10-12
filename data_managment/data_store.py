import json
import os

from const import (
    CLEANED_COMPLAINTS_PATH,
    CLEANED_RECALLS_PATH,
    DATA_REPORT_PATH,
    INTERM_RES,
    MATCHED_COMPLAINTS_PATH,
    RECALL_RISK_SUMMARY_PATH,
    VEHICLE_RISK_SUMMARY_PATH,
)


class DataStore:

    # Save cleaned, matched, and reporting intermediate data
    # ------------------------------------------------------
    def save_intermediate(self, complaints, recalls, matched, report):
        os.makedirs(INTERM_RES, exist_ok=True)
        complaints.to_parquet(CLEANED_COMPLAINTS_PATH, index=False)
        recalls.to_parquet(CLEANED_RECALLS_PATH, index=False)
        matched.to_parquet(MATCHED_COMPLAINTS_PATH, index=False)
        with open(DATA_REPORT_PATH, "w", encoding="utf-8") as report_file:
            json.dump(report, report_file, indent=2)

    # Save feature-dependent vehicle and recall risk summaries
    # --------------------------------------------------------
    def save_risk_summaries(self, vehicle_risk_summary, recall_risk_summary):
        os.makedirs(INTERM_RES, exist_ok=True)
        vehicle_risk_summary.to_parquet(VEHICLE_RISK_SUMMARY_PATH, index=False)
        recall_risk_summary.to_parquet(RECALL_RISK_SUMMARY_PATH, index=False)
