import pandas as pd


class DataCleaner:

    _COMPLAINT_RENAMES = {
        "MFR_NAME": "manufacturer",
        "MAKETXT": "make",
        "MODELTXT": "model",
        "YEARTXT": "model_year",
        "FAILDATE": "incident_date",
        "DATEA": "date_added",
        "LDATE": "complaint_date",
        "PURCH_DT": "purchase_date",
        "MANUF_DT": "manufacture_date",
        "CDESCR": "complaint_narrative",
    }

    _RECALL_RENAMES = {
        "NHTSA ID": "nhtsa_id",
        "DOCUMENT NAME": "document_name",
        "MAKE": "make",
        "MODEL": "model",
        "MODEL YEAR": "model_year",
        "SUMMARY": "summary",
    }

    _RECALL_DEDUPLICATION_COLUMNS = [
        "nhtsa_id",
        "document_name",
        "make",
        "model",
        "model_year",
    ]

    # Normalize identifiers used for vehicle matching
    # -----------------------------------------------
    def normalize_text(self, values):
        return (
            values.astype("string")
            .str.strip()
            .str.upper()
            .str.replace(r"[^A-Z0-9]+", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    # Clean and standardize complaint records
    # ----------------------------------------
    def clean_complaints(self, complaints):
        complaints = complaints.rename(columns=self._COMPLAINT_RENAMES).copy()
        complaints["manufacturer"] = self.normalize_text(
            complaints["manufacturer"]
        )
        complaints["make"] = self.normalize_text(complaints["make"])
        complaints["model"] = self.normalize_text(complaints["model"])
        complaints["model_year"] = pd.to_numeric(
            complaints["model_year"], errors="coerce"
        ).astype("Int64")

        for column in self.complaint_date_columns():
            complaints[column] = pd.to_datetime(
                complaints[column], format="%Y%m%d", errors="coerce"
            )

        return complaints

    # Clean, standardize, and deduplicate recall documents
    # ----------------------------------------------------
    def clean_recalls(self, recalls):
        recalls = recalls.rename(columns=self._RECALL_RENAMES).copy()
        recalls["make"] = self.normalize_text(recalls["make"])
        recalls["manufacturer"] = recalls["make"]
        recalls["model"] = self.normalize_text(recalls["model"])
        recalls["model_year"] = pd.to_numeric(
            recalls["model_year"], errors="coerce"
        ).astype("Int64")

        return recalls.drop_duplicates(
            subset=self._RECALL_DEDUPLICATION_COLUMNS,
            keep="last",
        ).reset_index(drop=True)

    # Return standardized complaint date columns
    # -------------------------------------------
    def complaint_date_columns(self):
        return [
            "incident_date",
            "date_added",
            "complaint_date",
            "purchase_date",
            "manufacture_date",
        ]

    # Return the recall deduplication key
    # -----------------------------------
    def recall_deduplication_columns(self):
        return self._RECALL_DEDUPLICATION_COLUMNS.copy()
